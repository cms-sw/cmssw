#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"

using namespace cms::cuda;

template <typename T>
struct RS {
  using type = std::uniform_int_distribution<T>;
  static auto ud() { return type(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); }
  static constexpr T imax = std::numeric_limits<T>::max();
};

template <>
struct RS<float> {
  using T = float;
  using type = std::uniform_real_distribution<float>;
  static auto ud() { return type(-std::numeric_limits<T>::max() / 2, std::numeric_limits<T>::max() / 2); }
  //  static auto ud() { return type(0,std::numeric_limits<T>::max()/2);}
  static constexpr int imax = std::numeric_limits<int>::max();
};

// A templated unsigned integer type with N bytes
template <int N>
struct uintN;

template <>
struct uintN<8> {
  using type = uint8_t;
};

template <>
struct uintN<16> {
  using type = uint16_t;
};

template <>
struct uintN<32> {
  using type = uint32_t;
};

template <>
struct uintN<64> {
  using type = uint64_t;
};

template <int N>
using uintN_t = typename uintN<N>::type;

// A templated unsigned integer type with the same size as T
template <typename T>
using uintT_t = uintN_t<sizeof(T) * 8>;

// Keep only the `N` most significant bytes of `t`, and set the others to zero
template <int N, typename T, typename SFINAE = std::enable_if_t<N <= sizeof(T)>>
void truncate(T& t) {
  const int shift = 8 * (sizeof(T) - N);
  union {
    T t;
    uintT_t<T> u;
  } c;
  c.t = t;
  c.u = c.u >> shift << shift;
  t = c.t;
}

template <typename T, int NS = sizeof(T), typename U = T, typename LL = long long>
void go(bool useShared) {
  std::mt19937 eng;
  //std::mt19937 eng2;
  auto rgen = RS<T>::ud();

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  constexpr int blocks = 10;
  constexpr int blockSize = 256 * 32;
  constexpr int N = blockSize * blocks;
  T v[N];
  uint16_t ind[N];

  constexpr bool sgn = T(-1) < T(0);
  std::cout << "Will sort " << N << (sgn ? " signed" : " unsigned")
            << (std::numeric_limits<T>::is_integer ? " 'ints'" : " 'float'") << " of size " << sizeof(T) << " using "
            << NS << " significant bytes" << std::endl;

  for (int i = 0; i < 50; ++i) {
    if (i == 49) {
      for (long long j = 0; j < N; j++)
        v[j] = 0;
    } else if (i > 30) {
      for (long long j = 0; j < N; j++)
        v[j] = rgen(eng);
    } else {
      uint64_t imax = (i < 15) ? uint64_t(RS<T>::imax) + 1LL : 255;
      for (uint64_t j = 0; j < N; j++) {
        v[j] = (j % imax);
        if (j % 2 && i % 2)
          v[j] = -v[j];
      }
    }

    uint32_t offsets[blocks + 1];
    offsets[0] = 0;
    for (int j = 1; j < blocks + 1; ++j) {
      offsets[j] = offsets[j - 1] + blockSize - 3 * j;
      assert(offsets[j] <= N);
    }

    if (i == 1) {  // special cases...
      offsets[0] = 0;
      offsets[1] = 0;
      offsets[2] = 19;
      offsets[3] = 32 + offsets[2];
      offsets[4] = 123 + offsets[3];
      offsets[5] = 256 + offsets[4];
      offsets[6] = 311 + offsets[5];
      offsets[7] = 2111 + offsets[6];
      offsets[8] = 256 * 11 + offsets[7];
      offsets[9] = 44 + offsets[8];
      offsets[10] = 3297 + offsets[9];
    }

    std::shuffle(v, v + N, eng);

    auto v_d = cms::cuda::make_device_unique<U[]>(N, nullptr);
    auto ind_d = cms::cuda::make_device_unique<uint16_t[]>(N, nullptr);
    auto ws_d = cms::cuda::make_device_unique<uint16_t[]>(N, nullptr);
    auto off_d = cms::cuda::make_device_unique<uint32_t[]>(blocks + 1, nullptr);

    cudaCheck(cudaMemcpy(v_d.get(), v, N * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(off_d.get(), offsets, 4 * (blocks + 1), cudaMemcpyHostToDevice));

    if (i < 2)
      std::cout << "launch for " << offsets[blocks] << std::endl;

    auto ntXBl __attribute__((unused)) = 1 == i % 4 ? 256 : 256;

    delta -= (std::chrono::high_resolution_clock::now() - start);
    // The MaxSize is the max size we allow between offsets (i.e. biggest set to sort when using shared memory).
    constexpr int MaxSize = 256 * 32;
    if (useShared)
      cms::cuda::launch(radixSortMultiWrapper<U, NS>,
                        {blocks, ntXBl, MaxSize * 2 /* sizeof(uint16_t)  */},
                        v_d.get(),
                        ind_d.get(),
                        off_d.get(),
                        nullptr);
    else
      cms::cuda::launch(
          radixSortMultiWrapper2<U, NS>, {blocks, ntXBl}, v_d.get(), ind_d.get(), off_d.get(), ws_d.get());

    if (i == 0)
      std::cout << "done for " << offsets[blocks] << std::endl;

    cudaCheck(cudaMemcpy(ind, ind_d.get(), 2 * N, cudaMemcpyDeviceToHost));

    delta += (std::chrono::high_resolution_clock::now() - start);

    if (i == 0)
      std::cout << "done for " << offsets[blocks] << std::endl;

    if (32 == i) {
      std::cout << LL(v[ind[0]]) << ' ' << LL(v[ind[1]]) << ' ' << LL(v[ind[2]]) << std::endl;
      std::cout << LL(v[ind[3]]) << ' ' << LL(v[ind[10]]) << ' ' << LL(v[ind[blockSize - 1000]]) << std::endl;
      std::cout << LL(v[ind[blockSize / 2 - 1]]) << ' ' << LL(v[ind[blockSize / 2]]) << ' '
                << LL(v[ind[blockSize / 2 + 1]]) << std::endl;
    }
    for (int ib = 0; ib < blocks; ++ib) {
      std::set<uint16_t> inds;
      if (offsets[ib + 1] > offsets[ib])
        inds.insert(ind[offsets[ib]]);
      for (auto j = offsets[ib] + 1; j < offsets[ib + 1]; j++) {
        inds.insert(ind[j]);
        auto a = v + offsets[ib];
        auto k1 = a[ind[j]];
        auto k2 = a[ind[j - 1]];
        truncate<NS>(k1);
        truncate<NS>(k2);
        if (k1 < k2)
          std::cout << ib << " not ordered at " << ind[j] << " : " << a[ind[j]] << ' ' << a[ind[j - 1]] << std::endl;
      }
      if (!inds.empty()) {
        assert(0 == *inds.begin());
        assert(inds.size() - 1 == *inds.rbegin());
      }
      if (inds.size() != (offsets[ib + 1] - offsets[ib]))
        std::cout << "error " << i << ' ' << ib << ' ' << inds.size() << "!=" << (offsets[ib + 1] - offsets[ib])
                  << std::endl;
      assert(inds.size() == (offsets[ib + 1] - offsets[ib]));
    }
  }  // 50 times
  std::cout << "cuda computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 50.
            << " ms" << std::endl;
}

int main() {
  cms::cudatest::requireDevices();

  bool useShared = false;

  std::cout << "using Global memory" << std::endl;

  go<int8_t>(useShared);
  go<int16_t>(useShared);
  go<int32_t>(useShared);
  go<int32_t, 3>(useShared);
  go<int64_t>(useShared);
  go<float, 4, float, double>(useShared);
  go<float, 2, float, double>(useShared);

  go<uint8_t>(useShared);
  go<uint16_t>(useShared);
  go<uint32_t>(useShared);
  // go<uint64_t>(v);

  useShared = true;

  std::cout << "using Shared memory" << std::endl;

  go<int8_t>(useShared);
  go<int16_t>(useShared);
  go<int32_t>(useShared);
  go<int32_t, 3>(useShared);
  go<int64_t>(useShared);
  go<float, 4, float, double>(useShared);
  go<float, 2, float, double>(useShared);

  go<uint8_t>(useShared);
  go<uint16_t>(useShared);
  go<uint32_t>(useShared);
  // go<uint64_t>(v);

  return 0;
}
