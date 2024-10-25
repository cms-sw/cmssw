#ifndef HeterogeneousCore_AlpakaInterface_interface_CachingAllocator_h
#define HeterogeneousCore_AlpakaInterface_interface_CachingAllocator_h

#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

#include <tbb/concurrent_queue.h>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AlpakaServiceFwd.h"

// Inspired by cub::CachingDeviceAllocator

namespace cms::alpakatools {

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
    static_assert(alpaka::isDevice<Device>, "TDev should be an alpaka Device type.");
    static_assert(alpaka::isQueue<Queue>, "TQueue should be an alpaka Queue type.");
    static_assert(std::is_same_v<Device, alpaka::Dev<Queue>> or std::is_same_v<Device, alpaka::DevCpu>,
                  "The \"memory device\" type can either be the same as the \"synchronisation device\" type, or be the "
                  "host CPU.");

    struct BlockDescriptor {
      std::optional<Buffer> buffer;
      std::optional<Queue> queue;
      std::optional<Event> event;
      size_t bytes = 0;
      size_t requested = 0;
      unsigned int bin = 0;

      // the "synchronisation device" for this block
      auto device() { return alpaka::getDev(*queue); }
    };

    struct CachedBytes {
      size_t free;       // Total bytes freed and cached on this device
      size_t live;       // Total bytes currently in use on this device
      size_t requested;  // Total bytes requested and currently in use on this device
    };

    explicit CachingAllocator(
        Device const& device,
        AllocatorConfig const& config,
        bool reuseSameQueueAllocations,  // Reuse non-ready allocations if they are in the same queue as the new one;
                                         // this is safe only if all memory operations are scheduled in the same queue.
                                         // In particular, this is not safe if the memory will be accessed without using
                                         // any queue, like host memory accessed directly or with immediate operations.
        bool debug = false);

    ~CachingAllocator();

    // Return a copy of the cache allocation status, for monitoring purposes
    CachedBytes cacheStatus() const;

    // Allocate given number of bytes on the current device associated to given queue
    BlockDescriptor* allocate(size_t bytes, Queue queue);

    // Frees an allocation, potentially caching the buffer in the free store
    void free(BlockDescriptor* block);

  private:
    // Fill a memory buffer with the specified bye value.
    // If the underlying device is the host and the allocator is configured to support immediate
    // (non queue-ordered) operations, fill the memory synchronously using std::memset.
    // Otherwise, let the alpaka queue schedule the operation.
    //
    // This is not used for deallocation/caching, because the memory may still be in use until the
    // corresponding event is reached.
    void immediateOrAsyncMemset(Queue queue, Buffer buffer, uint8_t value);

    // Return the maximum amount of memory that should be cached on this device
    size_t cacheSize(size_t maxCachedBytes, double maxCachedFraction) const;

    // Return (bin, bin size)
    std::tuple<unsigned int, size_t> findBin(size_t bytes) const;

    bool tryReuseCachedBlock(BlockDescriptor& block);

    Buffer allocateBuffer(size_t bytes, Queue const& queue);

    void allocateNewBlock(BlockDescriptor& block);

    void freeAllCached();

    struct BlockList {
      tbb::concurrent_queue<BlockDescriptor*> blocks_;
    };

    Device device_;  // the device where the memory is allocated
    inline static const std::string deviceType_ = alpaka::core::demangled<Device>;

    // List of free allocation blocks, cached and potentially available for reuse, index by the block bin
    std::vector<BlockList> cachedBlocks_;

    std::atomic<size_t> totalFree_ = 0;       // Total bytes freed and cached on this device
    std::atomic<size_t> totalLive_ = 0;       // Total bytes currently in use on this device
    std::atomic<size_t> totalRequested_ = 0;  // Total bytes requested and currently in use on this device

    const unsigned int binGrowth_;  // Geometric growth factor for bin-sizes
    const unsigned int minBin_;
    const unsigned int maxBin_;

    const size_t minBinBytes_;
    const size_t maxBinBytes_;
    const size_t maxCachedBytes_;  // Maximum aggregate cached bytes per device

    const bool reuseSameQueueAllocations_;
    const bool debug_;

    const bool fillAllocations_;
    const uint8_t fillAllocationValue_;
    const bool fillReallocations_;
    const uint8_t fillReallocationValue_;
    const bool fillDeallocations_;
    const uint8_t fillDeallocationValue_;
    const bool fillCaches_;
    const uint8_t fillCacheValue_;
  };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  extern template class CachingAllocator<alpaka_cuda_async::Device, alpaka_cuda_async::Queue>;
  extern template class CachingAllocator<alpaka_common::DevHost, alpaka_cuda_async::Queue>;
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  extern template class CachingAllocator<alpaka_rocm_async::Device, alpaka_rocm_async::Queue>;
  extern template class CachingAllocator<alpaka_common::DevHost, alpaka_rocm_async::Queue>;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  extern template class CachingAllocator<alpaka_serial_sync::Device, alpaka_serial_sync::Queue>;
  extern template class CachingAllocator<alpaka_common::DevHost, alpaka_serial_sync::Queue>;
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  extern template class CachingAllocator<alpaka_tbb_async::Device, alpaka_tbb_async::Queue>;
  extern template class CachingAllocator<alpaka_common::DevHost, alpaka_tbb_async::Queue>;
#endif

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_CachingAllocator_h
