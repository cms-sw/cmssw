#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AlpakaServiceFwd.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"

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

    // format a memory size in B/KiB/MiB/GiB/TiB
    inline std::string as_bytes(size_t value) {
      if (value == std::numeric_limits<size_t>::max()) {
        return "unlimited";
      } else if (value >= (1ul << 40) and value % (1ul << 40) == 0) {
        return std::to_string(value >> 40) + " TiB";
      } else if (value >= (1ul << 30) and value % (1ul << 30) == 0) {
        return std::to_string(value >> 30) + " GiB";
      } else if (value >= (1ul << 20) and value % (1ul << 20) == 0) {
        return std::to_string(value >> 20) + " MiB";
      } else if (value >= (1ul << 10) and value % (1ul << 10) == 0) {
        return std::to_string(value >> 10) + " KiB";
      } else {
        return std::to_string(value) + "   B";
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

  template <typename TDev, typename TQueue>
  CachingAllocator<TDev, TQueue>::CachingAllocator(
      Device const& device,
      AllocatorConfig const& config,
      bool reuseSameQueueAllocations,  // Reuse non-ready allocations if they are in the same queue as the new one;
                                       // this is safe only if all memory operations are scheduled in the same queue.
                                       // In particular, this is not safe if the memory will be accessed without using
                                       // any queue, like host memory accessed directly or with immediate operations.
      bool debug)
      : device_(device),
        cachedBlocks_(config.maxBin + 1),
        binGrowth_(config.binGrowth),
        minBin_(config.minBin),
        maxBin_(config.maxBin),
        minBinBytes_(detail::power(binGrowth_, minBin_)),
        maxBinBytes_(detail::power(binGrowth_, maxBin_)),
        maxCachedBytes_(cacheSize(config.maxCachedBytes, config.maxCachedFraction)),
        reuseSameQueueAllocations_(reuseSameQueueAllocations),
        debug_(debug),
        fillAllocations_(config.fillAllocations),
        fillAllocationValue_(config.fillAllocationValue),
        fillReallocations_(config.fillReallocations),
        fillReallocationValue_(config.fillReallocationValue),
        fillDeallocations_(config.fillDeallocations),
        fillDeallocationValue_(config.fillDeallocationValue),
        fillCaches_(config.fillCaches),
        fillCacheValue_(config.fillCacheValue) {
    if (debug_) {
      std::ostringstream out;
      out << "CachingAllocator settings\n"
          << "  bin growth " << binGrowth_ << "\n"
          << "  min bin    " << minBin_ << "\n"
          << "  max bin    " << maxBin_ << "\n"
          << "  resulting bins:\n";
      for (auto bin = minBin_; bin <= maxBin_; ++bin) {
        auto binSize = detail::power(binGrowth_, bin);
        out << "    " << std::right << std::setw(12) << detail::as_bytes(binSize) << '\n';
      }
      out << "  maximum amount of cached memory: " << detail::as_bytes(maxCachedBytes_);
      std::cout << out.str() << std::endl;
    }
  }

  template <typename TDev, typename TQueue>
  CachingAllocator<TDev, TQueue>::~CachingAllocator() {
    // this should never be called while some memory blocks are still live
    assert(totalLive_ == 0);
    freeAllCached();
  }

  // return a copy of the cache allocation status, for monitoring purposes
  template <typename TDev, typename TQueue>
  typename CachingAllocator<TDev, TQueue>::CachedBytes CachingAllocator<TDev, TQueue>::cacheStatus() const {
    return CachedBytes{totalFree_, totalLive_, totalRequested_};
  }

  // Fill a memory buffer with the specified byte value.
  // If the underlying device is the host and the allocator is configured to support immediate
  // (non queue-ordered) operations, fill the memory synchronously using std::memset.
  // Otherwise, let the alpaka queue schedule the operation.
  //
  // This is not used for deallocation/caching, because the memory may still be in use until the
  // corresponding event is reached.
  template <typename TDev, typename TQueue>
  void CachingAllocator<TDev, TQueue>::immediateOrAsyncMemset(Queue queue, Buffer buffer, uint8_t value) {
    // host-only
    if (std::is_same_v<Device, alpaka::DevCpu> and not reuseSameQueueAllocations_) {
      std::memset(buffer.data(), value, alpaka::getExtentProduct(buffer) * sizeof(alpaka::Elem<Buffer>));
    } else {
      alpaka::memset(queue, buffer, value);
    }
  }

  // Allocate given number of bytes on the current device associated to given queue
  template <typename TDev, typename TQueue>
  typename CachingAllocator<TDev, TQueue>::BlockDescriptor* CachingAllocator<TDev, TQueue>::allocate(size_t bytes,
                                                                                                     Queue queue) {
    // create a block descriptor for the requested allocation
    BlockDescriptor* block = new BlockDescriptor;
    block->queue = std::move(queue);
    block->requested = bytes;
    std::tie(block->bin, block->bytes) = findBin(bytes);

    // try to re-use a cached block, or allocate a new buffer
    if (tryReuseCachedBlock(*block)) {
      // fill the re-used memory block with a pattern
      if (fillReallocations_) {
        immediateOrAsyncMemset(*block->queue, *block->buffer, fillReallocationValue_);
      } else if (fillAllocations_) {
        immediateOrAsyncMemset(*block->queue, *block->buffer, fillAllocationValue_);
      }
    } else {
      allocateNewBlock(*block);
      // fill the newly allocated memory block with a pattern
      if (fillAllocations_) {
        immediateOrAsyncMemset(*block->queue, *block->buffer, fillAllocationValue_);
      }
    }

    return block;
  }

  // frees an allocation
  template <typename TDev, typename TQueue>
  void CachingAllocator<TDev, TQueue>::free(typename CachingAllocator<TDev, TQueue>::BlockDescriptor* block) {
    assert(block);
    assert(block->buffer);
    assert(block->buffer->data());
    assert(block->queue);
    assert(block->queue->m_spQueueImpl.get());
    assert(block->event);
    assert(block->event->m_spEventImpl.get());
    totalLive_ -= block->bytes;
    totalRequested_ -= block->requested;

    bool recache = (totalFree_ + block->bytes <= maxCachedBytes_);
    if (recache) {
      // If enqueuing the event fails, very likely an error has
      // occurred in the asynchronous processing. In that case the
      // error will show up in all device API function calls, and
      // the free() will be called by destructors during stack
      // unwinding. In order to avoid terminate() being called
      // because of multiple exceptions it is best to ignore these
      // errors.
      try {
        // fill memory blocks with a pattern before caching them
        if (fillCaches_) {
          alpaka::memset(*block->queue, *block->buffer, fillCacheValue_);
        } else if (fillDeallocations_) {
          alpaka::memset(*block->queue, *block->buffer, fillDeallocationValue_);
        }
        // record in the block a marker associated to the work queue
        alpaka::enqueue(*(block->queue), *(block->event));
      } catch (std::exception& e) {
        if (debug_) {
          std::ostringstream out;
          out << "CachingAllocator::free() caught an alpaka error: " << e.what() << "\n";
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes at "
              << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
              << block->event->m_spEventImpl.get() << ".\n";
          std::cout << out.str() << std::endl;
        }
        delete block;
        return;
      }
      totalFree_ += block->bytes;

      if (debug_) {
        std::ostringstream out;
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " returned " << block->bytes << " bytes at "
            << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
            << block->event->m_spEventImpl.get() << ".\n";
        std::cout << out.str() << std::endl;
      }

      // move the block into the free list
      auto& bin = cachedBlocks_[block->bin];
      assert(block);
      assert(block->buffer);
      assert(block->buffer->data());
      assert(block->queue);
      assert(block->queue->m_spQueueImpl.get());
      assert(block->event);
      assert(block->event->m_spEventImpl.get());
      bin.blocks_.push(block);
      block = nullptr;
    } else {
      // If the memset fails, very likely an error has occurred in the
      // asynchronous processing. In that case the error will show up in all
      // device API function calls, and the free() will be called by
      // destructors during stack unwinding. In order to avoid terminate()
      // being called because of multiple exceptions it is best to ignore
      // these errors.
      try {
        // fill memory blocks with a pattern before freeing them
        if (fillDeallocations_) {
          alpaka::memset(*block->queue, *block->buffer, fillDeallocationValue_);
        }
      } catch (std::exception& e) {
        if (debug_) {
          std::ostringstream out;
          out << "CachingAllocator::free() caught an alpaka error: " << e.what() << "\n";
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes at "
              << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
              << block->event->m_spEventImpl.get() << ".\n";
          std::cout << out.str() << std::endl;
        }
        return;
      }

      if (debug_) {
        std::ostringstream out;
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes at "
            << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
            << block->event->m_spEventImpl.get() << ".\n";
        std::cout << out.str() << std::endl;
      }

      // the buffer is not recached, delete it and free the memory associated to it
      assert(block);
      assert(block->buffer);
      assert(block->buffer->data());
      assert(block->queue);
      assert(block->queue->m_spQueueImpl.get());
      assert(block->event);
      assert(block->event->m_spEventImpl.get());
      delete block;
      block = nullptr;
    }
  }

  // return the maximum amount of memory that should be cached on this device
  template <typename TDev, typename TQueue>
  size_t CachingAllocator<TDev, TQueue>::cacheSize(size_t maxCachedBytes, double maxCachedFraction) const {
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
  template <typename TDev, typename TQueue>
  std::tuple<unsigned int, size_t> CachingAllocator<TDev, TQueue>::findBin(size_t bytes) const {
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

  template <typename TDev, typename TQueue>
  bool CachingAllocator<TDev, TQueue>::tryReuseCachedBlock(CachingAllocator<TDev, TQueue>::BlockDescriptor& block) {
    auto& bin = cachedBlocks_[block.bin];
    auto& blocks = bin.blocks_;

    std::vector<BlockDescriptor*> temp;
    temp.reserve(16);

    bool found = false;
    BlockDescriptor* candidate = nullptr;
    while (not found and blocks.try_pop(candidate)) {
      assert(candidate);
      assert(candidate->buffer);
      assert(candidate->buffer->data());
      assert(candidate->queue);
      assert(candidate->queue->m_spQueueImpl.get());
      assert(candidate->event);
      assert(candidate->event->m_spEventImpl.get());

      if ((reuseSameQueueAllocations_ and (*block.queue == *candidate->queue)) or
          alpaka::isComplete(*candidate->event)) {
        // reuse the block
        found = true;

        // transfer ownership of the old buffer
        block.buffer = std::move(candidate->buffer);
        assert(block.buffer);
        assert(block.buffer->data());

        // keep the queue used to request the allocation
        assert(block.queue);
        assert(block.queue->m_spQueueImpl.get());

        // if the new queue is on different device than the old event, create a new event, otherwise reuse the old event
        if (block.device() != alpaka::getDev(*(candidate->event))) {
          block.event = Event{block.device()};
        } else {
          if (debug_) {
            // only for debugging make a copy, so we can print the information of the old event
            block.event = candidate->event;
          } else {
            block.event = std::move(candidate->event);
          }
        }
        assert(block.event);
        assert(block.event->m_spEventImpl.get());

        // update the accounting information
        totalFree_ -= block.bytes;
        totalLive_ += block.bytes;
        totalRequested_ += block.requested;

        if (debug_) {
          std::ostringstream out;
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " reused cached block at "
              << block.buffer->data() << " (" << block.bytes << " bytes) for queue " << block.queue->m_spQueueImpl.get()
              << ", event " << block.event->m_spEventImpl.get() << " (previously associated with queue "
              << candidate->queue->m_spQueueImpl.get() << ", event " << candidate->event->m_spEventImpl.get() << ")."
              << std::endl;
          std::cout << out.str() << std::endl;
        }
      } else {
        // the candidate block is still busy, move it to the temporary buffer
        temp.push_back(candidate);
        candidate = nullptr;
      }
    }

    // either we found a block to reuse, or the free list is empty; stop looking and put the blocks in the temporary store back in the free list
    for (auto* tmp_block : temp) {
      blocks.push(tmp_block);
    }

    return found;
  }

  template <typename TDev, typename TQueue>
  typename CachingAllocator<TDev, TQueue>::Buffer CachingAllocator<TDev, TQueue>::allocateBuffer(size_t bytes,
                                                                                                 Queue const& queue) {
    if constexpr (std::is_same_v<Device, alpaka::Dev<Queue>>) {
      // allocate device memory
      return alpaka::allocBuf<std::byte, size_t>(device_, bytes);
    } else if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
      // allocate pinned host memory accessible by the queue's platform
      using Platform = alpaka::Platform<alpaka::Dev<Queue>>;
      return alpaka::allocMappedBuf<Platform, std::byte, size_t>(device_, platform<Platform>(), bytes);
    } else {
      // unsupported combination
      static_assert(std::is_same_v<Device, alpaka::Dev<Queue>> or std::is_same_v<Device, alpaka::DevCpu>,
                    "The \"memory device\" type can either be the same as the \"synchronisation device\" type, or be "
                    "the host CPU.");
    }
  }

  template <typename TDev, typename TQueue>
  void CachingAllocator<TDev, TQueue>::allocateNewBlock(CachingAllocator<TDev, TQueue>::BlockDescriptor& block) {
    try {
      block.buffer = allocateBuffer(block.bytes, *block.queue);
    } catch (std::runtime_error const& e) {
      // the allocation attempt failed: free all cached blocks on the device and retry
      if (debug_) {
        std::ostringstream out;
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " failed to allocate " << block.bytes
            << " bytes for queue " << block.queue->m_spQueueImpl.get() << ", retrying after freeing cached allocations"
            << std::endl;
        std::cout << out.str() << std::endl;
      }
      // TODO implement a method that frees only up to block.bytes bytes
      freeAllCached();

      // throw an exception if it fails again
      block.buffer = allocateBuffer(block.bytes, *block.queue);
    }

    // create a new event associated to the "synchronisation device"
    block.event = Event{block.device()};

    // update the statistics
    totalLive_ += block.bytes;
    totalRequested_ += block.requested;

    if (debug_) {
      std::ostringstream out;
      out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " allocated new block at "
          << block.buffer->data() << " (" << block.bytes << " bytes associated with queue "
          << block.queue->m_spQueueImpl.get() << ", event " << block.event->m_spEventImpl.get() << "." << std::endl;
      std::cout << out.str() << std::endl;
    }
  }

  template <typename TDev, typename TQueue>
  void CachingAllocator<TDev, TQueue>::freeAllCached() {
    for (auto& bin : cachedBlocks_) {
      // pops the blocks one at a time from the queue, and delete them
      BlockDescriptor* block = nullptr;
      while (bin.blocks_.try_pop(block)) {
        totalFree_ -= block->bytes;

        if (debug_) {
          std::ostringstream out;
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes.\n";
          std::cout << out.str() << std::endl;
        }

        delete block;
        block = nullptr;
      }
    }
  }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template class CachingAllocator<alpaka_cuda_async::Device, alpaka_cuda_async::Queue>;
  template class CachingAllocator<alpaka_common::DevHost, alpaka_cuda_async::Queue>;
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  template class CachingAllocator<alpaka_rocm_async::Device, alpaka_rocm_async::Queue>;
  template class CachingAllocator<alpaka_common::DevHost, alpaka_rocm_async::Queue>;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template class CachingAllocator<alpaka_serial_sync::Device, alpaka_serial_sync::Queue>;
  //template class CachingAllocator<alpaka_common::DevHost, alpaka_serial_sync::Queue>;  // skip: device and host are the same type
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  template class CachingAllocator<alpaka_tbb_async::Device, alpaka_tbb_async::Queue>;
  //template class CachingAllocator<alpaka_common::DevHost, alpaka_tbb_async::Queue>;  // skip: device and host are the same type
#endif

}  // namespace cms::alpakatools
