#ifndef HeterogeneousCore_AlpakaInterface_interface_CachingAllocator_h
#define HeterogeneousCore_AlpakaInterface_interface_CachingAllocator_h

#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AlpakaServiceFwd.h"

// Inspired by cub::CachingDeviceAllocator

namespace cms::alpakatools {

  namespace detail {

    inline constexpr unsigned int power(unsigned int base, unsigned int exponent) {
      unsigned int power = 1;
      while (exponent > 0) {
        if (exponent & 1) {
          power = power * base;
        }
        base = base * base;
        exponent = exponent >> 1;
      }
      return power;
    }

    // format a memory size in B/kB/MB/GB
    inline std::string as_bytes(size_t value) {
      if (value == std::numeric_limits<size_t>::max()) {
        return "unlimited";
      } else if (value >= (1 << 30) and value % (1 << 30) == 0) {
        return std::to_string(value >> 30) + " GB";
      } else if (value >= (1 << 20) and value % (1 << 20) == 0) {
        return std::to_string(value >> 20) + " MB";
      } else if (value >= (1 << 10) and value % (1 << 10) == 0) {
        return std::to_string(value >> 10) + " kB";
      } else {
        return std::to_string(value) + "  B";
      }
    }

  }  // namespace detail

  /*
   * The "memory device" identifies the memory space, i.e. the device where the memory is allocated.
   * A caching allocator object is associated to a single memory `Device`, set at construction time, and unchanged for
   * the lifetime of the allocator.
   *
   * Each allocation is associated to an event on a queue, that identifies the "synchronisation device" according to
   * which the synchronisation occurs.
   * The `Event` type depends only on the synchronisation `Device` type.
   * The `Queue` type depends on the synchronisation `Device` type and the queue properties, either `Sync` or `Async`.
   *
   * **Note**: how to handle different queue and event types in a single allocator ?  store and access type-punned
   * queues and events ?  or template the internal structures on them, but with a common base class ?
   * alpaka does rely on the compile-time type for dispatch.
   *
   * Common use case #1: accelerator's memory allocations
   *   - the "memory device" is the accelerator device (e.g. a GPU);
   *   - the "synchronisation device" is the same accelerator device;
   *   - the `Queue` type is usually always the same (either `Sync` or `Async`).
   *
   * Common use case #2: pinned host memory allocations
   *    - the "memory device" is the host device (e.g. system memory);
   *    - the "synchronisation device" is the accelerator device (e.g. a GPU) whose work queue will access the host;
   *      memory (direct memory access from the accelerator, or scheduling `alpaka::memcpy`/`alpaka::memset`), and can
   *      be different for each allocation;
   *    - the synchronisation `Device` _type_ could potentially be different, but memory pinning is currently tied to
   *      the accelerator's platform (CUDA, HIP, etc.), so the device type needs to be fixed to benefit from caching;
   *    - the `Queue` type can be either `Sync` _or_ `Async` on any allocation.
   */

  template <typename TDev,
            typename TQueue,
            typename = std::enable_if_t<alpaka::isDevice<TDev> and alpaka::isQueue<TQueue>>>
  class CachingAllocator {
  public:
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    friend class alpaka_cuda_async::AlpakaService;
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    friend class alpaka_rocm_async::AlpakaService;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    friend class alpaka_serial_sync::AlpakaService;
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    friend class alpaka_tbb_async::AlpakaService;
#endif

    using Device = TDev;                 // the "memory device", where the memory will be allocated
    using Queue = TQueue;                // the queue used to submit the memory operations
    using Event = alpaka::Event<Queue>;  // the events used to synchronise the operations
    using Buffer = alpaka::Buf<Device, std::byte, alpaka::DimInt<1u>, size_t>;

    // The "memory device" type can either be the same as the "synchronisation device" type, or be the host CPU.
    static_assert(std::is_same_v<Device, alpaka::Dev<Queue>> or std::is_same_v<Device, alpaka::DevCpu>,
                  "The \"memory device\" type can either be the same as the \"synchronisation device\" type, or be the "
                  "host CPU.");

    struct CachedBytes {
      size_t free = 0;       // total bytes freed and cached on this device
      size_t live = 0;       // total bytes currently in use oin this device
      size_t requested = 0;  // total bytes requested and currently in use on this device
    };

    explicit CachingAllocator(
        Device const& device,
        unsigned int binGrowth,          // bin growth factor;
        unsigned int minBin,             // smallest bin, corresponds to binGrowth^minBin bytes;
                                         // smaller allocations are rounded to this value;
        unsigned int maxBin,             // largest bin, corresponds to binGrowth^maxBin bytes;
                                         // larger allocations will fail;
        size_t maxCachedBytes,           // total storage for the allocator (0 means no limit);
        double maxCachedFraction,        // fraction of total device memory taken for the allocator (0 means no limit);
                                         // if both maxCachedBytes and maxCachedFraction are non-zero,
                                         // the smallest resulting value is used.
        bool reuseSameQueueAllocations,  // reuse non-ready allocations if they are in the same queue as the new one;
                                         // this is safe only if all memory operations are scheduled in the same queue
        bool debug)
        : device_(device),
          binGrowth_(binGrowth),
          minBin_(minBin),
          maxBin_(maxBin),
          minBinBytes_(detail::power(binGrowth, minBin)),
          maxBinBytes_(detail::power(binGrowth, maxBin)),
          maxCachedBytes_(cacheSize(maxCachedBytes, maxCachedFraction)),
          reuseSameQueueAllocations_(reuseSameQueueAllocations),
          debug_(debug) {
      if (debug_) {
        std::ostringstream out;
        out << "CachingAllocator settings\n"
            << "  bin growth " << binGrowth_ << "\n"
            << "  min bin    " << minBin_ << "\n"
            << "  max bin    " << maxBin_ << "\n"
            << "  resulting bins:\n";
        for (auto bin = minBin_; bin <= maxBin_; ++bin) {
          auto binSize = detail::power(binGrowth, bin);
          out << "    " << std::right << std::setw(12) << detail::as_bytes(binSize) << '\n';
        }
        out << "  maximum amount of cached memory: " << detail::as_bytes(maxCachedBytes_);
        std::cout << out.str() << std::endl;
      }
    }

    ~CachingAllocator() {
      {
        // this should never be called while some memory blocks are still live
        std::scoped_lock lock(mutex_);
        assert(liveBlocks_.empty());
        assert(cachedBytes_.live == 0);
      }

      freeAllCached();
    }

    // return a copy of the cache allocation status, for monitoring purposes
    CachedBytes cacheStatus() const {
      std::scoped_lock lock(mutex_);
      return cachedBytes_;
    }

    // Allocate given number of bytes on the current device associated to given queue
    void* allocate(size_t bytes, Queue queue) {
      // create a block descriptor for the requested allocation
      BlockDescriptor block;
      block.queue = std::move(queue);
      block.requested = bytes;
      std::tie(block.bin, block.bytes) = findBin(bytes);

      // try to re-use a cached block, or allocate a new buffer
      if (not tryReuseCachedBlock(block)) {
        allocateNewBlock(block);
      }

      return block.buffer->data();
    }

    // frees an allocation
    void free(void* ptr) {
      std::scoped_lock lock(mutex_);

      auto iBlock = liveBlocks_.find(ptr);
      if (iBlock == liveBlocks_.end()) {
        std::stringstream ss;
        ss << "Trying to free a non-live block at " << ptr;
        throw std::runtime_error(ss.str());
      }
      // remove the block from the list of live blocks
      BlockDescriptor block = std::move(iBlock->second);
      liveBlocks_.erase(iBlock);
      cachedBytes_.live -= block.bytes;
      cachedBytes_.requested -= block.requested;

      bool recache = (cachedBytes_.free + block.bytes <= maxCachedBytes_);
      if (recache) {
        alpaka::enqueue(*(block.queue), *(block.event));
        cachedBytes_.free += block.bytes;
        // after the call to insert(), cachedBlocks_ shares ownership of the buffer
        // TODO use std::move ?
        cachedBlocks_.insert(std::make_pair(block.bin, block));

        if (debug_) {
          std::ostringstream out;
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " returned " << block.bytes << " bytes at "
              << ptr << " from associated queue " << block.queue->m_spQueueImpl.get() << " , event "
              << block.event->m_spEventImpl.get() << " .\n\t\t " << cachedBlocks_.size() << " available blocks cached ("
              << cachedBytes_.free << " bytes), " << liveBlocks_.size() << " live blocks (" << cachedBytes_.live
              << " bytes) outstanding." << std::endl;
          std::cout << out.str() << std::endl;
        }
      } else {
        // if the buffer is not recached, it is automatically freed when block goes out of scope
        if (debug_) {
          std::ostringstream out;
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block.bytes << " bytes at "
              << ptr << " from associated queue " << block.queue->m_spQueueImpl.get() << ", event "
              << block.event->m_spEventImpl.get() << " .\n\t\t " << cachedBlocks_.size() << " available blocks cached ("
              << cachedBytes_.free << " bytes), " << liveBlocks_.size() << " live blocks (" << cachedBytes_.live
              << " bytes) outstanding." << std::endl;
          std::cout << out.str() << std::endl;
        }
      }
    }

  private:
    struct BlockDescriptor {
      std::optional<Buffer> buffer;
      std::optional<Queue> queue;
      std::optional<Event> event;
      size_t bytes = 0;
      size_t requested = 0;  // for monitoring only
      unsigned int bin = 0;

      // the "synchronisation device" for this block
      auto device() { return alpaka::getDev(*queue); }
    };

  private:
    // return the maximum amount of memory that should be cached on this device
    size_t cacheSize(size_t maxCachedBytes, double maxCachedFraction) const {
      // note that getMemBytes() returns 0 if the platform does not support querying the device memory
      size_t totalMemory = alpaka::getMemBytes(device_);
      size_t memoryFraction = static_cast<size_t>(maxCachedFraction * totalMemory);
      size_t size = std::numeric_limits<size_t>::max();
      if (maxCachedBytes > 0 and maxCachedBytes < size) {
        size = maxCachedBytes;
      }
      if (memoryFraction > 0 and memoryFraction < size) {
        size = memoryFraction;
      }
      return size;
    }

    // return (bin, bin size)
    std::tuple<unsigned int, size_t> findBin(size_t bytes) const {
      if (bytes < minBinBytes_) {
        return std::make_tuple(minBin_, minBinBytes_);
      }
      if (bytes > maxBinBytes_) {
        throw std::runtime_error("Requested allocation size " + std::to_string(bytes) +
                                 " bytes is too large for the caching detail with maximum bin " +
                                 std::to_string(maxBinBytes_) +
                                 " bytes. You might want to increase the maximum bin size");
      }
      unsigned int bin = minBin_;
      size_t binBytes = minBinBytes_;
      while (binBytes < bytes) {
        ++bin;
        binBytes *= binGrowth_;
      }
      return std::make_tuple(bin, binBytes);
    }

    bool tryReuseCachedBlock(BlockDescriptor& block) {
      std::scoped_lock lock(mutex_);

      // iterate through the range of cached blocks in the same bin
      const auto [begin, end] = cachedBlocks_.equal_range(block.bin);
      for (auto iBlock = begin; iBlock != end; ++iBlock) {
        if ((reuseSameQueueAllocations_ and (*block.queue == *(iBlock->second.queue))) or
            alpaka::isComplete(*(iBlock->second.event))) {
          // associate the cached buffer to the new queue
          auto queue = std::move(*(block.queue));
          // TODO cache (or remove) the debug information and use std::move()
          block = iBlock->second;
          block.queue = std::move(queue);

          // if the new queue is on different device than the old event, create a new event
          if (block.device() != alpaka::getDev(*(block.event))) {
            block.event = Event{block.device()};
          }

          // insert the cached block into the live blocks
          // TODO cache (or remove) the debug information and use std::move()
          liveBlocks_[block.buffer->data()] = block;

          // update the accounting information
          cachedBytes_.free -= block.bytes;
          cachedBytes_.live += block.bytes;
          cachedBytes_.requested += block.requested;

          if (debug_) {
            std::ostringstream out;
            out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " reused cached block at "
                << block.buffer->data() << " (" << block.bytes << " bytes) for queue "
                << block.queue->m_spQueueImpl.get() << ", event " << block.event->m_spEventImpl.get()
                << " (previously associated with queue " << iBlock->second.queue->m_spQueueImpl.get() << " , event "
                << iBlock->second.event->m_spEventImpl.get() << ")." << std::endl;
            std::cout << out.str() << std::endl;
          }

          // remove the reused block from the list of cached blocks
          cachedBlocks_.erase(iBlock);
          return true;
        }
      }

      return false;
    }

    Buffer allocateBuffer(size_t bytes, Queue const& queue) {
      if constexpr (std::is_same_v<Device, alpaka::Dev<Queue>>) {
        // allocate device memory
        return alpaka::allocBuf<std::byte, size_t>(device_, bytes);
      } else if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        // allocate pinned host memory accessible by the queue's platform
        return alpaka::allocMappedBuf<alpaka::Pltf<alpaka::Dev<Queue>>, std::byte, size_t>(device_, bytes);
      } else {
        // unsupported combination
        static_assert(std::is_same_v<Device, alpaka::Dev<Queue>> or std::is_same_v<Device, alpaka::DevCpu>,
                      "The \"memory device\" type can either be the same as the \"synchronisation device\" type, or be "
                      "the host CPU.");
      }
    }

    void allocateNewBlock(BlockDescriptor& block) {
      try {
        block.buffer = allocateBuffer(block.bytes, *block.queue);
      } catch (std::runtime_error const& e) {
        // the allocation attempt failed: free all cached blocks on the device and retry
        if (debug_) {
          std::ostringstream out;
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " failed to allocate " << block.bytes
              << " bytes for queue " << block.queue->m_spQueueImpl.get()
              << ", retrying after freeing cached allocations" << std::endl;
          std::cout << out.str() << std::endl;
        }
        // TODO implement a method that frees only up to block.bytes bytes
        freeAllCached();

        // throw an exception if it fails again
        block.buffer = allocateBuffer(block.bytes, *block.queue);
      }

      // create a new event associated to the "synchronisation device"
      block.event = Event{block.device()};

      {
        std::scoped_lock lock(mutex_);
        cachedBytes_.live += block.bytes;
        cachedBytes_.requested += block.requested;
        // TODO use std::move() ?
        liveBlocks_[block.buffer->data()] = block;
      }

      if (debug_) {
        std::ostringstream out;
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " allocated new block at "
            << block.buffer->data() << " (" << block.bytes << " bytes associated with queue "
            << block.queue->m_spQueueImpl.get() << ", event " << block.event->m_spEventImpl.get() << "." << std::endl;
        std::cout << out.str() << std::endl;
      }
    }

    void freeAllCached() {
      std::scoped_lock lock(mutex_);

      while (not cachedBlocks_.empty()) {
        auto iBlock = cachedBlocks_.begin();
        cachedBytes_.free -= iBlock->second.bytes;

        if (debug_) {
          std::ostringstream out;
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << iBlock->second.bytes
              << " bytes.\n\t\t  " << (cachedBlocks_.size() - 1) << " available blocks cached (" << cachedBytes_.free
              << " bytes), " << liveBlocks_.size() << " live blocks (" << cachedBytes_.live << " bytes) outstanding."
              << std::endl;
          std::cout << out.str() << std::endl;
        }

        cachedBlocks_.erase(iBlock);
      }
    }

    // TODO replace with a tbb::concurrent_multimap ?
    using CachedBlocks = std::multimap<unsigned int, BlockDescriptor>;  // ordered by the allocation bin
    // TODO replace with a tbb::concurrent_map ?
    using BusyBlocks = std::map<void*, BlockDescriptor>;  // ordered by the address of the allocated memory

    inline static const std::string deviceType_ = alpaka::core::demangled<Device>;

    mutable std::mutex mutex_;
    Device device_;  // the device where the memory is allocated

    CachedBytes cachedBytes_;
    CachedBlocks cachedBlocks_;  // Set of cached device allocations available for reuse
    BusyBlocks liveBlocks_;      // map of pointers to the live device allocations currently in use

    const unsigned int binGrowth_;  // Geometric growth factor for bin-sizes
    const unsigned int minBin_;
    const unsigned int maxBin_;

    const size_t minBinBytes_;
    const size_t maxBinBytes_;
    const size_t maxCachedBytes_;  // Maximum aggregate cached bytes per device

    const bool reuseSameQueueAllocations_;
    const bool debug_;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_CachingAllocator_h
