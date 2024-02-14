#ifndef HeterogeneousCore_AlpakaInterface_interface_radixSort_h
#define HeterogeneousCore_AlpakaInterface_interface_radixSort_h

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace cms::alpakatools {

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void dummyReorder(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {}

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void reorderSigned(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    //move negative first...

    auto& firstNeg = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    firstNeg = a[ind[0]] < 0 ? 0 : size;
    alpaka::syncBlockThreads(acc);

    // find first negative
    for (auto idx : independent_group_elements(acc, size - 1)) {
      if ((a[ind[idx]] ^ a[ind[idx + 1]]) < 0) {
        firstNeg = idx + 1;
      }
    }

    alpaka::syncBlockThreads(acc);

    for (auto idx : independent_group_elements(acc, firstNeg, size)) {
      ind2[idx - firstNeg] = ind[idx];
    }
    alpaka::syncBlockThreads(acc);

    for (auto idx : independent_group_elements(acc, firstNeg)) {
      ind2[idx + size - firstNeg] = ind[idx];
    }
    alpaka::syncBlockThreads(acc);

    for (auto idx : independent_group_elements(acc, size)) {
      ind[idx] = ind2[idx];
    }
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void reorderFloat(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    //move negative first...

    auto& firstNeg = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    firstNeg = a[ind[0]] < 0 ? 0 : size;
    alpaka::syncBlockThreads(acc);

    // find first negative
    for (auto idx : independent_group_elements(acc, size - 1)) {
      if ((a[ind[idx]] ^ a[ind[idx + 1]]) < 0)
        firstNeg = idx + 1;
    }
    alpaka::syncBlockThreads(acc);

    for (auto idx : independent_group_elements(acc, firstNeg, size)) {
      ind2[size - idx - 1] = ind[idx];
    }
    alpaka::syncBlockThreads(acc);

    for (auto idx : independent_group_elements(acc, firstNeg)) {
      ind2[idx + size - firstNeg] = ind[idx];
    }
    alpaka::syncBlockThreads(acc);

    for (auto idx : independent_group_elements(acc, size)) {
      ind[idx] = ind2[idx];
    }
  }

  // Radix sort implements a bytewise lexicographic order on the input data.
  // Data is reordered into bins indexed by the byte considered. But considering least significant bytes first
  // and respecting the existing order when binning the values, we achieve the lexicographic ordering.
  // The number of bytes actually considered is a parameter template parameter.
  // The post processing reorder
  // function fixes the order when bitwise ordering is not the order for the underlying type (only based on
  // most significant bit for signed types, integer or floating point).
  // The floating point numbers are reinterpret_cast into integers in the calling wrapper
  // This algorithm requires to run in a single block
  template <typename TAcc,
            typename T,   // shall be integer, signed or not does not matter here
            int NS,       // number of significant bytes to use in sorting.
            typename RF>  // The post processing reorder function.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radixSortImpl(
      const TAcc& acc, T const* __restrict__ a, uint16_t* ind, uint16_t* ind2, uint32_t size, RF reorder) {
    if constexpr (!requires_single_thread_per_block_v<TAcc>) {
      const auto warpSize = alpaka::warp::getSize(acc);
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      [[maybe_unused]] const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);
      // we expect a power of 2 here
      assert(warpSize && (0 == (warpSize & (warpSize - 1))));
      const std::size_t warpMask = warpSize - 1;

      // Define the bin size (d=8 => 1 byte bin).
      constexpr int binBits = 8, dataBits = 8 * sizeof(T), totalSortingPassses = dataBits / binBits;
      // Make sure the slices are data aligned
      static_assert(0 == dataBits % binBits);
      // Make sure the NS parameter makes sense
      static_assert(NS > 0 && NS <= sizeof(T));
      constexpr int binsNumber = 1 << binBits;
      constexpr int binsMask = binsNumber - 1;
      // Prefix scan iterations. NS is counted in full bytes and not slices.
      constexpr int initialSortingPass = int(sizeof(T)) - NS;

      // Count/index for the prefix scan
      // TODO: rename
      auto& c = alpaka::declareSharedVar<int32_t[binsNumber], __COUNTER__>(acc);
      // Temporary storage for prefix scan. Only really needed for first-of-warp keeping
      // Then used for thread to bin mapping TODO: change type to byte and remap to
      auto& ct = alpaka::declareSharedVar<int32_t[binsNumber], __COUNTER__>(acc);
      // Bin to thread index mapping (used to store the highest thread index within a bin number
      // batch of threads.
      // TODO: currently initialized to an invalid value, but could also be initialized to the
      // lowest possible value (change to bytes?)
      auto& cu = alpaka::declareSharedVar<int32_t[binsNumber], __COUNTER__>(acc);
      // TODO we could also have an explicit caching of the current index for each thread.

      // TODO: do those have to be shared?
      auto& ibs = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      auto& currentSortingPass = alpaka::declareSharedVar<int, __COUNTER__>(acc);

      ALPAKA_ASSERT_ACC(size > 0);
      // TODO: is this a hard requirement?
      ALPAKA_ASSERT_ACC(blockDimension >= binsNumber);

      currentSortingPass = initialSortingPass;

      auto j = ind;
      auto k = ind2;

      // Initializer index order to trivial increment.
      for (auto idx : independent_group_elements(acc, size)) {
        j[idx] = idx;
      }
      alpaka::syncBlockThreads(acc);

      // Iterate on the slices of the data.
      while (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, (currentSortingPass < totalSortingPassses))) {
        for (auto idx : independent_group_elements(acc, binsNumber)) {
          c[idx] = 0;
        }
        alpaka::syncBlockThreads(acc);
        const auto sortingPassShift = binBits * currentSortingPass;

        // fill bins (count elements in each bin)
        for (auto idx : independent_group_elements(acc, size)) {
          auto bin = (a[j[idx]] >> sortingPassShift) & binsMask;
          alpaka::atomicAdd(acc, &c[bin], 1, alpaka::hierarchy::Threads{});
        }
        alpaka::syncBlockThreads(acc);

        if (!threadIdxLocal && 1 == alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]) {
          //          printf("Pass=%d, Block=%d, ", currentSortingPass - 1, alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]);
          size_t total = 0;
          for (int i = 0; i < (int)binsNumber; i++) {
            //            printf("count[%d]=%d ", i, c[i] );
            total += c[i];
          }
          //          printf("total=%zu\n", total);
          assert(total == size);
        }
        // prefix scan "optimized"???...
        // TODO: we might be able to reuse the warpPrefixScan function
        // Warp level prefix scan
        for (auto idx : independent_group_elements(acc, binsNumber)) {
          auto x = c[idx];
          auto laneId = idx & warpMask;

          for (int offset = 1; offset < warpSize; offset <<= 1) {
            auto y = alpaka::warp::shfl(acc, x, laneId - offset);
            if (laneId >= (uint32_t)offset)
              x += y;
          }
          ct[idx] = x;
        }
        alpaka::syncBlockThreads(acc);

        // Block level completion of prefix scan (add last sum of each preceding warp)
        for (auto idx : independent_group_elements(acc, binsNumber)) {
          auto ss = (idx / warpSize) * warpSize - 1;
          c[idx] = ct[idx];
          for (int i = ss; i > 0; i -= warpSize)
            c[idx] += ct[i];
        }
        // Post prefix scan, c[bin] contains the offsets in index counts to the last index +1 for each bin

        /*
        //prefix scan for the nulls  (for documentation)
        if (threadIdxLocal==0)
          for (int i = 1; i < sb; ++i) c[i] += c[i-1];
        */

        // broadcast: we will fill the new index array downward, from offset c[bin], with one thread per
        // bin, working on one set of bin size elements at a time.
        // This will reorder the indices by the currently considered slice, otherwise preserving the previous order.
        ibs = size - 1;
        alpaka::syncBlockThreads(acc);

        // Iterate on bin-sized slices to (size - 1) / binSize + 1 iterations
        while (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, ibs >= 0)) {
          // Init
          for (auto idx : independent_group_elements(acc, binsNumber)) {
            cu[idx] = -1;
            ct[idx] = -1;
          }
          alpaka::syncBlockThreads(acc);

          // Find the highest index for all the threads dealing with a given bin (in cu[])
          // Also record the bin for each thread (in ct[])
          for (auto idx : independent_group_elements(acc, binsNumber)) {
            int i = ibs - idx;
            int32_t bin = -1;
            if (i >= 0) {
              bin = (a[j[i]] >> sortingPassShift) & binsMask;
              ct[idx] = bin;
              alpaka::atomicMax(acc, &cu[bin], int(i), alpaka::hierarchy::Threads{});
            }
          }
          alpaka::syncBlockThreads(acc);

          // FIXME: we can slash a memory access.
          for (auto idx : independent_group_elements(acc, binsNumber)) {
            int i = ibs - idx;
            // Are we still in inside the data?
            if (i >= 0) {
              int32_t bin = ct[idx];
              // Are we the thread with the highest index (from previous pass)?
              if (cu[bin] == i) {
                // With the highest index, we are actually the lowest thread number. We will
                // work "on behalf of" the higher thread numbers (including ourselves)
                // No way around scanning and testing for bin in ct[otherThread] number to find the other threads
                for (int peerThreadIdx = idx; peerThreadIdx < binsNumber; peerThreadIdx++) {
                  if (ct[peerThreadIdx] == bin) {
                    k[--c[bin]] = j[ibs - peerThreadIdx];
                  }
                }
              }
            }
            /*
            int32_t bin = (i >= 0 ? ((a[j[i]] >> sortingPassShift) & binsMask) : -1);
            if (i >= 0 && i == cu[bin])  // ensure to keep them in order: only one thread per bin is active, rest is idle.
              // 
              for (int ii = idx; ii < sb; ++ii)
                if (ct[ii] == bin) {
                  auto oi = ii - idx;
                  // assert(i>=oi);if(i>=oi)
                  k[--c[bin]] = j[i - oi]; // i = ibs - idx, oi = ii - idx => i - oi = ibs - ii;
                }
            */
          }
          alpaka::syncBlockThreads(acc);

          if (threadIdxLocal == 0) {
            ibs -= binsNumber;
            // https://github.com/cms-patatrack/pixeltrack-standalone/pull/210
            // TODO: is this really needed?
            alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
          }
          alpaka::syncBlockThreads(acc);
        }

        /*
        // broadcast for the nulls  (for documentation)
        if (threadIdxLocal==0)
        for (int i=size-first-1; i>=0; i--) { // =blockDim.x) {
          auto bin = (a[j[i]] >> d*p)&(sb-1);
          auto ik = atomicSub(&c[bin],1);
          k[ik-1] = j[i];
        }
        */

        alpaka::syncBlockThreads(acc);
        ALPAKA_ASSERT_ACC(c[0] == 0);

        // swap (local, ok)
        auto t = j;
        j = k;
        k = t;

        const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
        if (threadIdxLocal == 0)
          ++currentSortingPass;
        alpaka::syncBlockThreads(acc);
      }

      if ((dataBits != 8) && (0 == (NS & 1)))
        ALPAKA_ASSERT_ACC(j ==
                          ind);  // dataBits/binBits is even so ind is correct (the result is in the right location)

      // TODO this copy is (doubly?) redundant with the reorder
      if (j != ind)  // odd number of sorting passes, we need to move the result to the right array (ind[])
        for (auto idx : independent_group_elements(acc, size)) {
          ind[idx] = ind2[idx];
        };

      alpaka::syncBlockThreads(acc);

      // now move negative first... (if signed)
      // TODO: the ind2 => ind copy should have beed deferred. We should pass (j != ind) as an extra parameter
      reorder(acc, a, ind, ind2, size);
    } else {
      //static_assert(false);
    }
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<std::is_unsigned<T>::value && !requires_single_thread_per_block_v<TAcc>, T>::type* =
                nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    radixSortImpl<TAcc, T, NS>(acc, a, ind, ind2, size, dummyReorder<TAcc, T>);
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value &&
                                        !requires_single_thread_per_block_v<TAcc>,
                                    T>::type* = nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    radixSortImpl<TAcc, T, NS>(acc, a, ind, ind2, size, reorderSigned<TAcc, T>);
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<std::is_floating_point<T>::value && !requires_single_thread_per_block_v<TAcc>,
                                    T>::type* = nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    static_assert(sizeof(T) == sizeof(int), "radixSort with the wrong type size");
    using I = int;
    radixSortImpl<TAcc, I, NS>(acc, (I const*)(a), ind, ind2, size, reorderFloat<TAcc, I>);
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<requires_single_thread_per_block_v<TAcc>, T>::type* = nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    static_assert(requires_single_thread_per_block_v<TAcc>, "CPU sort (not a radixSort) called wtth wrong accelerator");
    // Initialize the index array
    std::iota(ind, ind + size, 0);
    /*
    printf("std::stable_sort(a=%p, ind=%p, indmax=%p, size=%d)\n", a, ind, ind + size, size);
    for (uint32_t i=0; i<10 && i<size; i++) {
      printf ("a[%d]=%ld ", i, (long int)a[i]);
    }
    printf("\n");
    for (uint32_t i=0; i<10 && i<size; i++) {
      printf ("ind[%d]=%d ", i, ind[i]);
    }
    printf("\n");
    */
    std::stable_sort(ind, ind + size, [a](uint16_t i0, uint16_t i1) { return a[i0] < a[i1]; });
    /*
    for (uint32_t i=0; i<10 && i<size; i++) {
      printf ("ind[%d]=%d ", i, ind[i]);
    }
    printf("\n");
    */
  }

  template <typename TAcc, typename T, int NS = sizeof(T)>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void radixSortMulti(
      const TAcc& acc, T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
    // TODO: check
    // Sort multiple blocks of data in v[] separated by in chunks located at offsets[]
    // extern __shared__ uint16_t ws[];
    uint16_t* ws = alpaka::getDynSharedMem<uint16_t>(acc);

    const uint32_t blockIdx(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
    auto a = v + offsets[blockIdx];
    auto ind = index + offsets[blockIdx];
    auto ind2 = nullptr == workspace ? ws : workspace + offsets[blockIdx];
    auto size = offsets[blockIdx + 1] - offsets[blockIdx];
    assert(offsets[blockIdx + 1] >= offsets[blockIdx]);
    if (size > 0)
      radixSort<TAcc, T, NS>(acc, a, ind, ind2, size);
  }

  template <typename T, int NS = sizeof(T)>
  struct radixSortMultiWrapper {
    /* We cannot set launch_bounds in alpaka, so both kernel wrappers are identical (keeping CUDA/HIP code for reference for the moment)
#if defined(__CUDACC__) || defined(__HIPCC__)
    //__global__ void __launch_bounds__(256, 4)
#endif
*/
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  T const* v,
                                  uint16_t* index,
                                  uint32_t const* offsets,
                                  uint16_t* workspace,
                                  size_t sharedMemBytes = 0) const {
      radixSortMulti<TAcc, T, NS>(acc, v, index, offsets, workspace);
    }
  };

  template <typename T, int NS = sizeof(T)>
  struct radixSortMultiWrapper2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  T const* v,
                                  uint16_t* index,
                                  uint32_t const* offsets,
                                  uint16_t* workspace,
                                  size_t sharedMemBytes = 0) const {
      radixSortMulti<TAcc, T, NS>(acc, v, index, offsets, workspace);
    }
  };
}  // namespace cms::alpakatools

namespace alpaka::trait {
  // specialize the BlockSharedMemDynSizeBytes trait to specify the amount of
  // block shared dynamic memory for the radixSortMultiWrapper kernel
  template <typename TAcc, typename T, int NS>
  struct BlockSharedMemDynSizeBytes<cms::alpakatools::radixSortMultiWrapper<T, NS>, TAcc> {
    // the size in bytes of the shared memory allocated for a block
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
        cms::alpakatools::radixSortMultiWrapper<T, NS> const& /* kernel */,
        alpaka_common::Vec1D /* threads */,
        alpaka_common::Vec1D /* elements */,
        T const* /* v */,
        uint16_t* /* index */,
        uint32_t const* /* offsets */,
        uint16_t* workspace,
        size_t sharedMemBytes) {
      if (workspace != nullptr)
        return 0;
      /* The shared memory workspace is 'blockspace * 2' in CUDA *but that's a value coincidence... TODO: check */
      //printf ("in BlockSharedMemDynSizeBytes<cms::alpakatools::radixSortMultiWrapper<T, NS>, TAcc>: shared mem size = %d\n", (int)sharedMemBytes);
      return sharedMemBytes;
    }
  };
}  // namespace alpaka::trait

#endif  // HeterogeneousCore_AlpakaInterface_interface_radixSort_h
