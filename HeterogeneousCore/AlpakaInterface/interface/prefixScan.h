#ifndef HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
#define HeterogeneousCore_AlpakaInterface_interface_prefixScan_h

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
namespace cms::alpakatools {
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr bool isPowerOf2(T v) {
    // returns true iif v has only one bit set.
    while (v) {
      if (v & 1)
        return !(v >> 1);
      else
        v >>= 1;
    }
    return false;
  }

  template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(
      const TAcc& acc, int32_t laneId, T const* ci, T* co, uint32_t i, bool active = true) {
    // ci and co may be the same
    T x = active ? ci[i] : 0;
    CMS_UNROLL_LOOP
    for (int32_t offset = 1; offset < alpaka::warp::getSize(acc); offset <<= 1) {
      // Force the exact type for integer types otherwise the compiler will find the template resolution ambiguous.
      using dataType = std::conditional_t<std::is_floating_point_v<T>, T, std::int32_t>;
      T y = alpaka::warp::shfl(acc, static_cast<dataType>(x), laneId - offset);
      if (laneId >= offset)
        x += y;
    }
    if (active)
      co[i] = x;
  }

  template <typename TAcc, typename T, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(
      const TAcc& acc, int32_t laneId, T* c, uint32_t i, bool active = true) {
    warpPrefixScan(acc, laneId, c, c, i, active);
  }

  // limited to warpSize² elements
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void blockPrefixScan(
      const TAcc& acc, T const* ci, T* co, int32_t size, T* ws = nullptr) {
    if constexpr (!requires_single_thread_per_block_v<TAcc>) {
      const auto warpSize = alpaka::warp::getSize(acc);
      int32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      int32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_ACC(ws);
      ALPAKA_ASSERT_ACC(size <= warpSize * warpSize);
      ALPAKA_ASSERT_ACC(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
      ALPAKA_ASSERT_ACC(isPowerOf2(warpSize));
      auto laneId = blockThreadIdx & (warpSize - 1);
      auto warpUpRoundedSize = (size + warpSize - 1) / warpSize * warpSize;

      for (auto i = first; i < warpUpRoundedSize; i += blockDimension) {
        // When padding the warp, warpPrefixScan is a noop
        warpPrefixScan(acc, laneId, ci, co, i, i < size);
        if (i < size) {
          // Skipped in warp padding threads.
          auto warpId = i / warpSize;
          ALPAKA_ASSERT_ACC(warpId < warpSize);
          if ((warpSize - 1) == laneId)
            ws[warpId] = co[i];
        }
      }
      alpaka::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(acc, laneId, ws, blockThreadIdx);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        int32_t warpId = i / warpSize;
        co[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
    } else {
      co[0] = ci[0];
      for (int32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
    }
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void blockPrefixScan(const TAcc& acc,
                                                           T* __restrict__ c,
                                                           int32_t size,
                                                           T* __restrict__ ws = nullptr) {
    if constexpr (!requires_single_thread_per_block_v<TAcc>) {
      const auto warpSize = alpaka::warp::getSize(acc);
      int32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      int32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_ACC(ws);
      ALPAKA_ASSERT_ACC(size <= warpSize * warpSize);
      ALPAKA_ASSERT_ACC(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
      auto laneId = blockThreadIdx & (warpSize - 1);
      auto warpUpRoundedSize = (size + warpSize - 1) / warpSize * warpSize;

      for (auto i = first; i < warpUpRoundedSize; i += blockDimension) {
        // When padding the warp, warpPrefixScan is a noop
        warpPrefixScan(acc, laneId, c, i, i < size);
        if (i < size) {
          // Skipped in warp padding threads.
          auto warpId = i / warpSize;
          ALPAKA_ASSERT_ACC(warpId < warpSize);
          if ((warpSize - 1) == laneId)
            ws[warpId] = c[i];
        }
      }
      alpaka::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(acc, laneId, ws, blockThreadIdx);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        auto warpId = i / warpSize;
        c[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
    } else {
      for (int32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
    }
  }

  // in principle not limited....
  template <typename T>
  struct multiBlockPrefixScan {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, T const* ci, T* co, uint32_t size, int32_t numBlocks, int32_t* pc, std::size_t warpSize) const {
      // Get shared variable. The workspace is needed only for multi-threaded accelerators.
      T* ws = nullptr;
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        ws = alpaka::getDynSharedMem<T>(acc);
      }
      ALPAKA_ASSERT_ACC(warpSize == static_cast<std::size_t>(alpaka::warp::getSize(acc)));
      [[maybe_unused]] const auto elementsPerGrid = alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u];
      const auto elementsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
      const auto threadsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const auto blocksPerGrid = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      ALPAKA_ASSERT_ACC(elementsPerGrid >= size);
      // first each block does a scan
      [[maybe_unused]] int off = elementsPerBlock * blockIdx;
      if (size - off > 0) {
        blockPrefixScan(acc, ci + off, co + off, std::min(elementsPerBlock, size - off), ws);
      }

      // count blocks that finished
      auto& isLastBlockDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
      //__shared__ bool isLastBlockDone;
      if (0 == threadIdx) {
        alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
        auto value = alpaka::atomicAdd(acc, pc, 1, alpaka::hierarchy::Blocks{});  // block counter
        isLastBlockDone = (value == (int(blocksPerGrid) - 1));
      }

      alpaka::syncBlockThreads(acc);

      if (!isLastBlockDone)
        return;

      ALPAKA_ASSERT_ACC(int(blocksPerGrid) == *pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block except the last, which receives 0.
      T* psum = nullptr;
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        psum = ws + warpSize;
      } else {
        psum = alpaka::getDynSharedMem<T>(acc);
      }
      for (int32_t i = threadIdx, ni = blocksPerGrid; i < ni; i += threadsPerBlock) {
        auto j = elementsPerBlock * i + elementsPerBlock - 1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      alpaka::syncBlockThreads(acc);
      blockPrefixScan(acc, psum, psum, blocksPerGrid, ws);

      // now it would have been handy to have the other blocks around...
      // Simplify the computation by having one version where threads per block = block size
      // and a second for the one thread per block accelerator.
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        //  Here threadsPerBlock == elementsPerBlock
        for (uint32_t i = threadIdx + threadsPerBlock, k = 0; i < size; i += threadsPerBlock, ++k) {
          co[i] += psum[k];
        }
      } else {
        // We are single threaded here, adding partial sums starting with the 2nd block.
        for (uint32_t i = elementsPerBlock; i < size; i++) {
          co[i] += psum[i / elementsPerBlock - 1];
        }
      }
    }
  };
}  // namespace cms::alpakatools

// declare the amount of block shared memory used by the multiBlockPrefixScan kernel
namespace alpaka::trait {
  // Variable size shared mem
  template <typename TAcc, typename T>
  struct BlockSharedMemDynSizeBytes<cms::alpakatools::multiBlockPrefixScan<T>, TAcc> {
    template <typename TVec>
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
        cms::alpakatools::multiBlockPrefixScan<T> const& /* kernel */,
        TVec const& /* blockThreadExtent */,
        TVec const& /* threadElemExtent */,
        T const* /* ci */,
        T const* /* co */,
        int32_t /* size */,
        int32_t numBlocks,
        int32_t const* /* pc */,
        // This trait function does not receive the accelerator object to look up the warp size
        std::size_t warpSize) {
      // We need workspace (T[warpsize]) + partial sums (T[numblocks]).
      if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
        return sizeof(T) * numBlocks;
      } else {
        return sizeof(T) * (warpSize + numBlocks);
      }
    }
  };

}  // namespace alpaka::trait

#endif  // HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
