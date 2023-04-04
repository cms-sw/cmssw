#ifndef AlpakaCore_radixSort_h
#define AlpakaCore_radixSort_h

#include <cstdint>
#include <type_traits>
#include <alpaka/alpaka.hpp>

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
    for_each_element_in_block_strided(acc, size - 1, [&](uint32_t idx) {
      if ((a[ind[idx]] ^ a[ind[idx + 1]]) < 0)
        firstNeg = idx + 1;
    });

    alpaka::syncBlockThreads(acc);

    for_each_element_in_block_strided(acc, size, firstNeg, [&](uint32_t idx) { ind2[idx - firstNeg] = ind[idx]; });
    alpaka::syncBlockThreads(acc);

    for_each_element_in_block_strided(acc, firstNeg, [&](uint32_t idx) { ind2[idx + size - firstNeg] = ind[idx]; });
    alpaka::syncBlockThreads(acc);

    for_each_element_in_block_strided(acc, size, [&](uint32_t idx) { ind[idx] = ind2[idx]; });
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void reorderFloat(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    //move negative first...

    auto& firstNeg = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);
    firstNeg = a[ind[0]] < 0 ? 0 : size;
    alpaka::syncBlockThreads(acc);

    // find first negative
    for_each_element_in_block_strided(acc, size - 1, [&](uint32_t idx) {
      if ((a[ind[idx]] ^ a[ind[idx + 1]]) < 0)
        firstNeg = idx + 1;
    });
    alpaka::syncBlockThreads(acc);

    for_each_element_in_block_strided(acc, size, firstNeg, [&](uint32_t idx) { ind2[size - idx - 1] = ind[idx]; });
    alpaka::syncBlockThreads(acc);

    for_each_element_in_block_strided(acc, firstNeg, [&](uint32_t idx) { ind2[idx + size - firstNeg] = ind[idx]; });
    alpaka::syncBlockThreads(acc);

    for_each_element_in_block_strided(acc, size, [&](uint32_t idx) { ind[idx] = ind2[idx]; });
  }

  template <typename TAcc,
            typename T,  // shall be interger
            int NS,      // number of significant bytes to use in sorting
            typename RF>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSortImpl(
      const TAcc& acc, T const* __restrict__ a, uint16_t* ind, uint16_t* ind2, uint32_t size, RF reorder) {
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    const uint32_t blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u]);

    constexpr int d = 8, w = 8 * sizeof(T);
    constexpr int sb = 1 << d;
    constexpr int ps = int(sizeof(T)) - NS;

    auto& c = alpaka::declareSharedVar<int32_t[sb], __COUNTER__>(acc);
    auto& ct = alpaka::declareSharedVar<int32_t[sb], __COUNTER__>(acc);
    auto& cu = alpaka::declareSharedVar<int32_t[sb], __COUNTER__>(acc);
    auto& ibs = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    auto& p = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    ALPAKA_ASSERT_OFFLOAD(size > 0);
    ALPAKA_ASSERT_OFFLOAD(blockDimension >= sb);

    p = ps;

    auto j = ind;
    auto k = ind2;

    for_each_element_in_block_strided(acc, size, [&](uint32_t idx) { j[idx] = idx; });
    alpaka::syncBlockThreads(acc);

    while (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, (p < w / d))) {
      for_each_element_in_block_strided(acc, sb, [&](uint32_t idx) { c[idx] = 0; });
      alpaka::syncBlockThreads(acc);

      // fill bins
      for_each_element_in_block_strided(acc, size, [&](uint32_t idx) {
        auto bin = (a[j[idx]] >> d * p) & (sb - 1);
        alpaka::atomicAdd(acc, &c[bin], 1, alpaka::hierarchy::Threads{});
      });
      alpaka::syncBlockThreads(acc);

      // prefix scan "optimized"???...
      for_each_element_in_block(acc, sb, [&](uint32_t idx) {
        auto x = c[idx];
        auto laneId = idx & 0x1f;

        for (int offset = 1; offset < 32; offset <<= 1) {
#if defined(__CUDA_ARCH__)
          auto y = __shfl_up_sync(0xffffffff, x, offset);
#elif defined(__HIP_DEVICE_COMPILE__)
          auto y = __shfl_up(x, offset);
#endif
          if (laneId >= (uint32_t)offset)
            x += y;
        }
        ct[idx] = x;
      });
      alpaka::syncBlockThreads(acc);

      for_each_element_in_block(acc, sb, [&](uint32_t idx) {
        auto ss = (idx / 32) * 32 - 1;
        c[idx] = ct[idx];
        for (int i = ss; i > 0; i -= 32)
          c[idx] += ct[i];
      });

      /*
      //prefix scan for the nulls  (for documentation)
      if (threadIdxLocal==0)
        for (int i = 1; i < sb; ++i) c[i] += c[i-1];
      */

      // broadcast
      ibs = size - 1;
      alpaka::syncBlockThreads(acc);

      while (alpaka::syncBlockThreadsPredicate<alpaka::BlockAnd>(acc, ibs > 0)) {
        for_each_element_in_block(acc, sb, [&](uint32_t idx) {
          cu[idx] = -1;
          ct[idx] = -1;
        });
        alpaka::syncBlockThreads(acc);

        for_each_element_in_block(acc, sb, [&](uint32_t idx) {
          int i = ibs - idx;
          int32_t bin = -1;
          if (i >= 0) {
            bin = (a[j[i]] >> d * p) & (sb - 1);
            ct[idx] = bin;
            alpaka::atomicMax(acc, &cu[bin], int(i), alpaka::hierarchy::Threads{});
          }
        });
        alpaka::syncBlockThreads(acc);

        for_each_element_in_block(acc, sb, [&](uint32_t idx) {
          int i = ibs - idx;
          int32_t bin = (i >= 0 ? ((a[j[i]] >> d * p) & (sb - 1)) : -1);
          if (i >= 0 && i == cu[bin])  // ensure to keep them in order
            for (int ii = idx; ii < sb; ++ii)
              if (ct[ii] == bin) {
                auto oi = ii - idx;
                // assert(i>=oi);if(i>=oi)
                k[--c[bin]] = j[i - oi];
              }
        });
        alpaka::syncBlockThreads(acc);

        if (threadIdxLocal == 0) {
          ibs -= sb;
          // cms-patatrack/pixeltrack-standalone#210
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
      ALPAKA_ASSERT_OFFLOAD(c[0] == 0);

      // swap (local, ok)
      auto t = j;
      j = k;
      k = t;

      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      if (threadIdxLocal == 0)
        ++p;
      alpaka::syncBlockThreads(acc);
    }

    if ((w != 8) && (0 == (NS & 1)))
      ALPAKA_ASSERT_OFFLOAD(j == ind);  // w/d is even so ind is correct

    if (j != ind)  // odd...
      for_each_element_in_block_strided(acc, size, [&](uint32_t idx) { ind[idx] = ind2[idx]; });

    alpaka::syncBlockThreads(acc);

    // now move negative first... (if signed)
    reorder(acc, a, ind, ind2, size);
#endif
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<std::is_unsigned<T>::value, T>::type* = nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    radixSortImpl<TAcc, T, NS>(acc, a, ind, ind2, size, dummyReorder<TAcc, T>);
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    radixSortImpl<TAcc, T, NS>(acc, a, ind, ind2, size, reorderSigned<TAcc, T>);
  }

  template <typename TAcc,
            typename T,
            int NS = sizeof(T),  // number of significant bytes to use in sorting
            typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSort(
      const TAcc& acc, T const* a, uint16_t* ind, uint16_t* ind2, uint32_t size) {
    static_assert(sizeof(T) == sizeof(int), "radixSort with the wrong type size");
    using I = int;
    radixSortImpl<TAcc, I, NS>(acc, (I const*)(a), ind, ind2, size, reorderFloat<TAcc, I>);
  }

  /* Not needed
template <typename T, int NS = sizeof(T)>
ALPAKA_FN_ACC ALPAKA_FN_INLINE __attribute__((always_inline)) void radixSortMulti(T const* v,
                                               uint16_t* index,
                                               uint32_t const* offsets,
                                               uint16_t* workspace) {

  extern __shared__ uint16_t ws[];

  auto a = v + offsets[blockIdx.x];
  auto ind = index + offsets[blockIdx.x];
  auto ind2 = nullptr == workspace ? ws : workspace + offsets[blockIdx.x];
  auto size = offsets[blockIdx.x + 1] - offsets[blockIdx.x];
  assert(offsets[blockIdx.x + 1] >= offsets[blockIdx.x]);
  if (size > 0)
    radixSort<T, NS>(a, ind, ind2, size);
}

    template <typename T, int NS = sizeof(T)>
    __global__ void __launch_bounds__(256, 4)
        radixSortMultiWrapper(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
      radixSortMulti<T, NS>(v, index, offsets, workspace);
    }

    template <typename T, int NS = sizeof(T)>
    __global__ void radixSortMultiWrapper2(T const* v, uint16_t* index, uint32_t const* offsets, uint16_t* workspace) {
      radixSortMulti<T, NS>(v, index, offsets, workspace);
    }
*/

}  // namespace cms::alpakatools

#endif  // AlpakaCore_radixSort_h
