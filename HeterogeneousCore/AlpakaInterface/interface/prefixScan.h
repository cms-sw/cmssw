#ifndef HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
#define HeterogeneousCore_AlpakaInterface_interface_prefixScan_h

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
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

  template <alpaka::concepts::Acc TAcc, typename T>
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

  template <alpaka::concepts::Acc TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(
      const TAcc& acc, int32_t laneId, T* c, uint32_t i, bool active = true) {
    warpPrefixScan(acc, laneId, c, c, i, active);
  }

  // limited to warpSize² elements
  template <alpaka::concepts::Acc TAcc, typename T>
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

  template <alpaka::concepts::Acc TAcc, typename T>
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

  // Throws an exception with a message containing the shared memory requirements and limit.
  void throwSharedMemoryLimitExceeded(const size_t nElements,
                                      const uint32_t nBlocks,
                                      const size_t requiredSharedMem,
                                      const size_t sharedMemLimit);

  // Verify shared memory requirements
  template <alpaka::concepts::Acc TAcc, typename TSize>
  ALPAKA_FN_INLINE static void checkSharedMemoryPrefixScan(TSize nElements,
                                                           uint32_t nBlocks,
                                                           alpaka::Dev<TAcc> const& device) {
    auto requiredSharedMem = (nBlocks + alpaka::getPreferredWarpSize(device)) * sizeof(TSize);
    auto sharedMemLimit = alpaka::getAccDevProps<TAcc>(device).m_sharedMemSizeBytes;
    if (requiredSharedMem > sharedMemLimit) {
      throwSharedMemoryLimitExceeded(static_cast<size_t>(nElements), nBlocks, requiredSharedMem, sharedMemLimit);
    }
  }

  // in principle not limited.... in practice limited by shared memory size and occupancy.
  template <typename T>
  struct multiBlockPrefixScan {
    template <alpaka::concepts::Acc TAcc>
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
        blockPrefixScan(acc, ci + off, co + off, alpaka::math::min(acc, elementsPerBlock, size - off), ws);
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
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        if (blocksPerGrid <= warpSize * warpSize)
          blockPrefixScan(acc, psum, blocksPerGrid, ws);
        else {
          auto off = 0u;
          while (off + warpSize * warpSize < blocksPerGrid) {
            blockPrefixScan(acc, psum + off, warpSize * warpSize, ws);
            off = off + warpSize * warpSize - 1;
            // ^ this -1 is to keep the previous round total sum around
            alpaka::syncBlockThreads(acc);
          }
          blockPrefixScan(acc, psum + off, psum + off, blocksPerGrid - off, ws);
        }
      } else {
        blockPrefixScan(acc, psum, blocksPerGrid, ws);
      }
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

  // Two kernel approach, not shared-memory limited.
  // Kernel A: scan one level (tile per block) and emit one block sum per block.
  // It is called recursively until the block sums array is reduced to one element, orchestration happens on host.
  // Kernel B: add scanned block offsets to each block (except block 0).

  // Kernel A
  template <typename T>
  struct scanTilesWriteBlockSums {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, T const* ci, T* co, uint32_t size, T* blockSums, std::size_t warpSize) const {
      T* ws = nullptr;
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        ws = alpaka::getDynSharedMem<T>(acc);
      }

      const auto elementsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
      const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];

      auto off = elementsPerBlock * blockIdx;
      if (off >= size)
        return;

      auto n = alpaka::math::min(acc, elementsPerBlock, size - off);
      blockPrefixScan(acc, ci + off, co + off, static_cast<int32_t>(n), ws);

      if (threadIdx == 0u) {
        blockSums[blockIdx] = co[off + n - 1];
      }
    }
  };

  // Kernel B
  template <typename T>
  struct addScannedBlockOffsets {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, T* data, uint32_t size, T const* scannedBlockSums) const {
      const auto elementsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
      const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];

      if (blockIdx == 0u)
        return;

      T const add = scannedBlockSums[blockIdx - 1u];
      uint32_t begin = elementsPerBlock * blockIdx;
      uint32_t end = alpaka::math::min(acc, begin + elementsPerBlock, size);
      for (auto i = begin + threadIdx; i < end; i += elementsPerBlock) {
        data[i] += add;
      }
    }
  };

  // Helper struct and function to compute the number of levels and block sizes for the two-kernel prefix scan.
  // 3 levels (0, 1, 2) can cover up to 1024^3 ~ 10^9 elements, enough for any realistic use case
  constexpr uint32_t iterativePrefixScanThreads = 1024u;
  constexpr uint32_t iterativePrefixScanMaxLevels = 3;
  struct PrefixScanLevelPlan {
    uint32_t nLevels = 0;
    std::vector<uint32_t> levelSize;
    std::vector<uint32_t> levelBlocks;
  };

  // Throws an exception with a message containing the requested iterations and the set compile-time limit
  void throwIterativePrefixScanMaxLevelsExceeded(const size_t nElements, const uint32_t nLevels);

  ALPAKA_FN_INLINE static PrefixScanLevelPlan makePrefixScanLevelPlan(uint32_t nElements) {
    PrefixScanLevelPlan p;
    if (nElements == 0u) {
      return p;
    }

    p.levelSize.reserve(iterativePrefixScanMaxLevels);
    p.levelBlocks.reserve(iterativePrefixScanMaxLevels);

    p.nLevels = 1u;
    p.levelSize.emplace_back(nElements);
    p.levelBlocks.emplace_back((nElements + iterativePrefixScanThreads - 1u) / iterativePrefixScanThreads);

    while (p.levelBlocks[p.nLevels - 1u] > 1u) {
      if (p.nLevels >= iterativePrefixScanMaxLevels) {
        throwIterativePrefixScanMaxLevelsExceeded(nElements, p.nLevels);
      }
      p.levelSize.emplace_back(p.levelBlocks[p.nLevels - 1u]);
      p.levelBlocks.emplace_back((p.levelSize[p.nLevels] + iterativePrefixScanThreads - 1u) /
                                 iterativePrefixScanThreads);
      ++p.nLevels;
    }
    return p;
  }

  template <alpaka::concepts::Acc TAcc, typename TQueue, typename T>
  ALPAKA_FN_INLINE static void iterativePrefixScan(T* input, T* output, uint32_t size, TQueue& queue) {
    if (size == 0u) {
      return;
    }

    if constexpr (!requires_single_thread_per_block_v<TAcc>) {
      auto const plan = makePrefixScanLevelPlan(size);
      assert(plan.nLevels > 0);

      std::vector<cms::alpakatools::device_buffer<alpaka::Dev<TQueue>, T[]>> blockSumsBuffers;
      blockSumsBuffers.reserve(plan.nLevels);
      for (uint32_t l = 0; l < plan.nLevels; ++l) {
        blockSumsBuffers.emplace_back(cms::alpakatools::make_device_buffer<T[]>(queue, plan.levelBlocks[l]));
      }

      auto const warpSize = alpaka::getPreferredWarpSize(alpaka::getDev(queue));

      // Kernel A on level-0 input data
      auto workDiv = cms::alpakatools::make_workdiv<TAcc>(plan.levelBlocks[0], iterativePrefixScanThreads);
      alpaka::exec<TAcc>(queue,
                         workDiv,
                         scanTilesWriteBlockSums<T>{},
                         input,
                         output,
                         plan.levelSize[0],
                         blockSumsBuffers[0].data(),
                         warpSize);

      // Iterative use of kernel A on block-sum levels
      for (uint32_t l = 1; l < plan.nLevels; ++l) {
        auto workDiv = cms::alpakatools::make_workdiv<TAcc>(plan.levelBlocks[l], iterativePrefixScanThreads);
        alpaka::exec<TAcc>(queue,
                           workDiv,
                           scanTilesWriteBlockSums<T>{},
                           blockSumsBuffers[l - 1].data(),
                           blockSumsBuffers[l - 1].data(),
                           plan.levelSize[l],
                           blockSumsBuffers[l].data(),
                           warpSize);
      }

      // Kernel B from top-1 down to level 0
      for (int32_t l = static_cast<int32_t>(plan.nLevels) - 2; l >= 0; --l) {
        auto workDiv = cms::alpakatools::make_workdiv<TAcc>(plan.levelBlocks[l], iterativePrefixScanThreads);
        T* levelData = (l == 0) ? output : blockSumsBuffers[l - 1].data();
        alpaka::exec<TAcc>(
            queue, workDiv, addScannedBlockOffsets<T>{}, levelData, plan.levelSize[l], blockSumsBuffers[l].data());
      }
    } else {
      output[0] = input[0];
      for (uint32_t i = 1; i < size; ++i) {
        output[i] = input[i] + output[i - 1];
      }
    }
  }

}  // namespace cms::alpakatools

// declare the amount of block shared memory used by the multiBlockPrefixScan kernel
namespace alpaka::trait {
  // Variable size shared mem
  template <alpaka::concepts::Acc TAcc, typename T>
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

  // Two-kernel approach requires only workspace for the first kernel, which is sized to the warp size.
  // Kernel A
  template <alpaka::concepts::Acc TAcc, typename T>
  struct BlockSharedMemDynSizeBytes<cms::alpakatools::scanTilesWriteBlockSums<T>, TAcc> {
    template <typename TVec>
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
        cms::alpakatools::scanTilesWriteBlockSums<T> const& /* kernel */,
        TVec const& /* blockThreadExtent */,
        TVec const& /* threadElemExtent */,
        T const* /* ci */,
        T const* /* co */,
        uint32_t /* size */,
        T* /* blockSums */,
        // This trait function does not receive the accelerator object to look up the warp size
        std::size_t warpSize) {
      if constexpr (cms::alpakatools::requires_single_thread_per_block_v<TAcc>) {
        return 0;
      } else {
        return sizeof(T) * warpSize;
      }
    }
  };

  // Kernel B does not require shared memory.
  template <alpaka::concepts::Acc TAcc, typename T>
  struct BlockSharedMemDynSizeBytes<cms::alpakatools::addScannedBlockOffsets<T>, TAcc> {
    template <typename TVec>
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
        cms::alpakatools::addScannedBlockOffsets<T> const&, TVec const&, TVec const&, T const*, uint32_t, T const*) {
      return 0;
    }
  };

}  // namespace alpaka::trait

#endif  // HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
