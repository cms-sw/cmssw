#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <syncstream>
#include <string>
#include <tuple>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AlpakaServiceFwd.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h"

// Originally inspired by the cub::CachingDeviceAllocator

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

    // Format a memory size in B/KiB/MiB/GiB/TiB.
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
   * See HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h for an explanation of the "memory device",
   * "synchronisation device", and of the reuseSameQueueAllocations parameter.
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
      std::osyncstream out(std::cerr);
      out << "CachingAllocator settings\n"
          << "  bin growth " << binGrowth_ << "\n"
          << "  min bin    " << minBin_ << "\n"
          << "  max bin    " << maxBin_ << "\n"
          << "  resulting bins:\n";
      for (auto bin = minBin_; bin <= maxBin_; ++bin) {
        auto binSize = detail::power(binGrowth_, bin);
        out << "    " << std::right << std::setw(12) << detail::as_bytes(binSize) << '\n';
      }
      out << "  maximum amount of cached memory: " << detail::as_bytes(maxCachedBytes_) << '\n';
    }
  }

  template <typename TDev, typename TQueue>
  CachingAllocator<TDev, TQueue>::~CachingAllocator() {
    // This should never be called while some memory blocks are still live.
    assert(totalLive_ == 0);
    freeAllCached();
  }

  // Return a copy of the cache allocation status, for monitoring purposes.
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
    // Host-only
    if (std::is_same_v<Device, alpaka::DevCpu> and not reuseSameQueueAllocations_) {
      std::memset(buffer.data(), value, alpaka::getExtentProduct(buffer) * sizeof(alpaka::Elem<Buffer>));
    } else {
      alpaka::memset(queue, buffer, value);
    }
  }

  // Allocate at least the requested number of bytes on the device associated to given queue.
  template <typename TDev, typename TQueue>
  typename CachingAllocator<TDev, TQueue>::BlockDescriptor* CachingAllocator<TDev, TQueue>::allocate(size_t bytes,
                                                                                                     Queue queue) {
    // Create a block descriptor for the requested allocation.
    auto block = std::make_unique<BlockDescriptor>();
    block->queue = std::move(queue);
    block->requested = bytes;
    std::tie(block->bin, block->bytes) = findBin(bytes);

    // Try to re-use a cached block, or allocate a new buffer.
    if (tryReuseCachedBlock(*block)) {
      // Fill the re-used memory block with a pattern.
      if (fillReallocations_) {
        immediateOrAsyncMemset(*block->queue, *block->buffer, fillReallocationValue_);
      } else if (fillAllocations_) {
        immediateOrAsyncMemset(*block->queue, *block->buffer, fillAllocationValue_);
      }
    } else {
      allocateNewBlock(*block);
      // Fill the newly allocated memory block with a pattern.
      if (fillAllocations_) {
        immediateOrAsyncMemset(*block->queue, *block->buffer, fillAllocationValue_);
      }
    }

    // Update the statistics.
    totalLive_ += block->bytes;
    totalRequested_ += block->requested;

    return block.release();
  }

  // Free an allocation, potentially caching the memory block for later reuse.
  template <typename TDev, typename TQueue>
  void CachingAllocator<TDev, TQueue>::free(typename CachingAllocator<TDev, TQueue>::BlockDescriptor* ptr) noexcept {
    // Take ownership of the allocation.
    std::unique_ptr<BlockDescriptor> block(ptr);
    assert(block);
    assert(block->buffer);
    assert(block->buffer->data());
    assert(block->queue);
    assert(block->queue->m_spQueueImpl.get());
    assert(block->event);
    assert(block->event->m_spEventImpl.get());

    // Update the statistics.
    totalLive_ -= block->bytes;
    totalRequested_ -= block->requested;

    bool recache = (totalFree_ + block->bytes <= maxCachedBytes_);
    if (recache) {
      // If the memset or enqueuing the event fail, very likely an error has
      // occurred in the asynchronous processing. In that case the error will
      // show up in all device API function calls, and free() will be called by
      // destructors during stack unwinding. In order to avoid terminate() being
      // called because of multiple exceptions it is best to ignore the errors.
      try {
        // Fill memory blocks with a pattern before caching them.
        if (fillCaches_) {
          alpaka::memset(*block->queue, *block->buffer, fillCacheValue_);
        } else if (fillDeallocations_) {
          alpaka::memset(*block->queue, *block->buffer, fillDeallocationValue_);
        }
        // Record in the block a marker associated to the work queue.
        alpaka::enqueue(*(block->queue), *(block->event));
      } catch (std::exception& e) {
        if (debug_) {
          std::osyncstream out(std::cerr);
          out << "CachingAllocator::free() caught an alpaka error: " << e.what() << "\n";
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes at "
              << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
              << block->event->m_spEventImpl.get() << ".\n";
        }
        // Free the block.
        // Note: the destructors of the underlying objects should not throw even
        // if the runtime is in an invalid state.
        block.reset();
        return;
      }
      totalFree_ += block->bytes;

      if (debug_) {
        std::osyncstream out(std::cerr);
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " returned " << block->bytes << " bytes at "
            << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
            << block->event->m_spEventImpl.get() << ".\n";
      }

      // Move the block into the free list.
      auto& bin = cachedBlocks_[block->bin];
      assert(block);
      assert(block->buffer);
      assert(block->buffer->data());
      assert(block->queue);
      assert(block->queue->m_spQueueImpl.get());
      assert(block->event);
      assert(block->event->m_spEventImpl.get());
      bin.blocks_.push(std::move(block));
      assert(not block);
    } else {
      // If the memset fails, very likely an error has occurred in the
      // asynchronous processing. In that case the error will show up in all
      // device API function calls, and free() will be called by destructors
      // during stack unwinding. In order to avoid terminate() being called
      // because of multiple exceptions it is best to ignore the errors.
      try {
        // Fill memory blocks with a pattern before freeing them.
        if (fillDeallocations_) {
          alpaka::memset(*block->queue, *block->buffer, fillDeallocationValue_);
        }
      } catch (std::exception& e) {
        if (debug_) {
          std::osyncstream out(std::cerr);
          out << "CachingAllocator::free() caught an alpaka error: " << e.what() << "\n";
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes at "
              << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
              << block->event->m_spEventImpl.get() << ".\n";
        }
        // Free the block.
        // Note: the destructors of the underlying objects should not throw even
        // if the runtime is in an invalid state.
        block.reset();
        return;
      }

      if (debug_) {
        std::osyncstream out(std::cerr);
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes at "
            << block->buffer->data() << " from associated queue " << block->queue->m_spQueueImpl.get() << ", event "
            << block->event->m_spEventImpl.get() << ".\n";
      }

      // The buffer is not recached, free it and the memory associated to it.
      assert(block);
      assert(block->buffer);
      assert(block->buffer->data());
      assert(block->queue);
      assert(block->queue->m_spQueueImpl.get());
      assert(block->event);
      assert(block->event->m_spEventImpl.get());
      block.reset();
    }
  }

  // Return the maximum amount of memory that should be cached on this device.
  template <typename TDev, typename TQueue>
  size_t CachingAllocator<TDev, TQueue>::cacheSize(size_t maxCachedBytes, double maxCachedFraction) const {
    // Note that getMemBytes() returns 0 if the platform does not support querying the device memory.
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

  // Return (bin, bin size).
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

    // Create a temporary storage to hold the blocks that are still busy while looking for one that is ready.
    // Most of the time this is not actually used, so delay the initial reserve to the first usage.
    std::vector<std::unique_ptr<BlockDescriptor>> temp;

    // Note that if an exception is thrown during the loop, the temporary storage will be destroyed and the blocks it
    // contain will be released.
    // As these are free blocks and not accounted in the totalLive_ value, this should not cause problems with the
    // assertion in the destructor of the CachingAllocator.

    bool found = false;
    std::unique_ptr<BlockDescriptor> candidate;
    while (not found and blocks.try_pop(candidate)) {
      assert(candidate);
      assert(candidate->buffer);
      assert(candidate->buffer->data());
      assert(candidate->queue);
      assert(candidate->queue->m_spQueueImpl.get());
      assert(candidate->event);
      assert(candidate->event->m_spEventImpl.get());

      // If the candidate block is still busy, move it to the temporary buffer and look for another block.
      if ((not reuseSameQueueAllocations_ or (*block.queue != *candidate->queue)) and
          not alpaka::isComplete(*candidate->event)) {
        // This is a simple heuristic based on a measurement performed using the reduced 2024 HLT menu (including only
        // the alpaka modules) running with 256 cores on a single GPU: using an initial capacity that is one quarter
        // of the original queue size, over 60% of the non-empty cases do not require any reallocation, while the other
        // will need at most two (except for the rare circumstances where the queue grows significantly while looking
        // for a ready block).
        // On average, this reserves 18 entries and requires 0.6 reallocations.
        // Note that after the first try_pop() the queue size is one less the original value.
        if (temp.empty()) {
          temp.reserve(blocks.unsafe_size() / 4 + 1);
        }
        temp.push_back(std::move(candidate));
        continue;
      }

      // Reuse the block.
      found = true;

      // Transfer ownership of the old buffer.
      block.buffer = std::move(candidate->buffer);
      assert(block.buffer);
      assert(block.buffer->data());

      // Update the statistics; totalLive_ and totalRequested_ are updated by allocate().
      totalFree_ -= block.bytes;

      // Keep the queue used to request the allocation.
      assert(block.queue);
      assert(block.queue->m_spQueueImpl.get());

      // If the new queue is on different device than the old event, create a
      // new event, otherwise reuse the old event.
      // Device memory allocators are per-device, so this can only happen for
      // a host allocator caching mapped memory buffers.
      if (block.device() != alpaka::getDev(*(candidate->event))) {
        block.event = Event{block.device()};
      } else {
        if (debug_) {
          // Only for debugging: make a copy to print the information of the old event.
          block.event = candidate->event;
        } else {
          block.event = std::move(candidate->event);
        }
      }
      assert(block.event);
      assert(block.event->m_spEventImpl.get());

      if (debug_) {
        std::osyncstream out(std::cerr);
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " reused cached block at "
            << block.buffer->data() << " (" << block.bytes << " bytes) for queue " << block.queue->m_spQueueImpl.get()
            << ", event " << block.event->m_spEventImpl.get() << " (previously associated with queue "
            << candidate->queue->m_spQueueImpl.get() << ", event " << candidate->event->m_spEventImpl.get() << ").\n";
      }

      // Free the block descriptor and the resources still associated to it.
      candidate.reset();
    }

    // Either we found a block to reuse, or the free list is empty; stop looking
    // and put the blocks in the temporary store back in the free list.
    for (auto& tmp_block : temp) {
      blocks.push(std::move(tmp_block));
    }

    return found;
  }

  template <typename TDev, typename TQueue>
  typename CachingAllocator<TDev, TQueue>::Buffer CachingAllocator<TDev, TQueue>::allocateBuffer(size_t bytes,
                                                                                                 Queue const& queue) {
    if constexpr (std::is_same_v<Device, alpaka::Dev<Queue>>) {
      // Allocate device memory.
      return alpaka::allocBuf<std::byte, size_t>(device_, bytes);
    } else if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
      // Allocate pinned host memory accessible by the queue's platform.
      using Platform = alpaka::Platform<alpaka::Dev<Queue>>;
      return alpaka::allocMappedBuf<std::byte, size_t>(device_, platform<Platform>(), bytes);
    } else {
      // Unsupported combination.
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
      // The allocation attempt failed: free all cached blocks on the device and retry
      if (debug_) {
        std::osyncstream out(std::cerr);
        out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " failed to allocate " << block.bytes
            << " bytes for queue " << block.queue->m_spQueueImpl.get()
            << ", retrying after freeing cached allocations.\n";
      }
      // TODO
      //   - implement a method that frees only up to block.bytes bytes ?
      //     that may not work, if the memory is fragmented;
      //   - try to reuse larger cached block ?
      //   - or, try to free a larger cached block and to allocate a buffer ?
      freeAllCached();

      // Let it throw an exception if it fails again.
      block.buffer = allocateBuffer(block.bytes, *block.queue);
    }

    // Create a new event associated to the "synchronisation device".
    block.event = Event{block.device()};

    // The statistics are updated by allocate().

    if (debug_) {
      std::osyncstream out(std::cerr);
      out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " allocated new block at "
          << block.buffer->data() << " (" << block.bytes << " bytes associated with queue "
          << block.queue->m_spQueueImpl.get() << ", event " << block.event->m_spEventImpl.get() << ".\n";
    }
  }

  template <typename TDev, typename TQueue>
  void CachingAllocator<TDev, TQueue>::freeAllCached() {
    for (auto& bin : cachedBlocks_) {
      // Pop the blocks one at a time from the queue, and free them.
      std::unique_ptr<BlockDescriptor> block;
      while (bin.blocks_.try_pop(block)) {
        totalFree_ -= block->bytes;
        if (debug_) {
          std::osyncstream out(std::cerr);
          out << "\t" << deviceType_ << " " << alpaka::getName(device_) << " freed " << block->bytes << " bytes.\n";
        }
        block.reset();
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
