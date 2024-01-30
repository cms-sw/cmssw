#include <algorithm>
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/radixSort.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

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
void go(Queue& queue, bool useShared) {
  std::mt19937 eng;
  //std::mt19937 eng2;
  auto rgen = RS<T>::ud();

  std::chrono::high_resolution_clock::duration delta = 0ns;
  constexpr int blocks = 10;
  constexpr int blockSize = 256 * 32;
  constexpr int N = blockSize * blocks;
  auto v_h = cms::alpakatools::make_host_buffer<T[]>(queue, N);
  //uint16_t ind_h[N];

  constexpr bool sgn = T(-1) < T(0);
  std::cout << "Will sort " << N << (sgn ? " signed" : " unsigned")
            << (std::numeric_limits<T>::is_integer ? " 'ints'" : " 'float'") << " of size " << sizeof(T) << " using "
            << NS << " significant bytes" << std::endl;

  for (int i = 0; i < 50; ++i) {
    if (i == 49) {
      for (long long j = 0; j < N; j++)
        v_h[j] = 0;
    } else if (i > 30) {
      for (long long j = 0; j < N; j++)
        v_h[j] = rgen(eng);
    } else {
      uint64_t imax = (i < 15) ? uint64_t(RS<T>::imax) + 1LL : 255;
      for (uint64_t j = 0; j < N; j++) {
        v_h[j] = (j % imax);
        if (j % 2 && i % 2)
          v_h[j] = -v_h[j];
      }
    }

    auto offsets_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, blocks + 1);
    offsets_h[0] = 0;
    for (int j = 1; j < blocks + 1; ++j) {
      offsets_h[j] = offsets_h[j - 1] + blockSize - 3 * j;
      assert(offsets_h[j] <= N);
    }

    if (i == 1) {  // special cases...
      offsets_h[0] = 0;
      offsets_h[1] = 0;
      offsets_h[2] = 19;
      offsets_h[3] = 32 + offsets_h[2];
      offsets_h[4] = 123 + offsets_h[3];
      offsets_h[5] = 256 + offsets_h[4];
      offsets_h[6] = 311 + offsets_h[5];
      offsets_h[7] = 2111 + offsets_h[6];
      offsets_h[8] = 256 * 11 + offsets_h[7];
      offsets_h[9] = 44 + offsets_h[8];
      offsets_h[10] = 3297 + offsets_h[9];
    }

    std::shuffle(v_h.data(), v_h.data() + N, eng);

    auto v_d = cms::alpakatools::make_device_buffer<U[]>(queue, N);
    auto ind_d = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, N);
    auto ind_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue, N);
    auto ws_d = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, N);
    auto off_d = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, blocks + 1);

    alpaka::memcpy(queue, v_d, v_h);
    alpaka::memcpy(queue, off_d, offsets_h);

    if (i < 2)
      std::cout << "launch for " << offsets_h[blocks] << std::endl;

    auto ntXBl = 1 == i % 4 ? 256 : 256;

    auto start = std::chrono::high_resolution_clock::now();
    // The MaxSize is the max size we allow between offsets (i.e. biggest set to sort when using shared memory).
    constexpr int MaxSize = 256 * 32;
    auto workdiv = make_workdiv<Acc1D>(blocks, ntXBl);
    if (useShared)
      // The original CUDA version used to call a kernel with __launch_bounds__(256, 4) specifier
      //
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workdiv,
                                                      radixSortMultiWrapper<U, NS>{},
                                                      v_d.data(),
                                                      ind_d.data(),
                                                      off_d.data(),
                                                      nullptr,
                                                      MaxSize * sizeof(uint16_t)));
    else
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1D>(
              workdiv, radixSortMultiWrapper2<U, NS>{}, v_d.data(), ind_d.data(), off_d.data(), ws_d.data()));

    if (i < 2)
      std::cout << "launch done for " << offsets_h[blocks] << std::endl;

    alpaka::memcpy(queue, ind_h, ind_d);
    alpaka::wait(queue);

    delta += std::chrono::high_resolution_clock::now() - start;

    if (i < 2)
      std::cout << "kernel and read back done for " << offsets_h[blocks] << std::endl;

    if (32 == i) {
      std::cout << LL(v_h[ind_h[0]]) << ' ' << LL(v_h[ind_h[1]]) << ' ' << LL(v_h[ind_h[2]]) << std::endl;
      std::cout << LL(v_h[ind_h[3]]) << ' ' << LL(v_h[ind_h[10]]) << ' ' << LL(v_h[ind_h[blockSize - 1000]])
                << std::endl;
      std::cout << LL(v_h[ind_h[blockSize / 2 - 1]]) << ' ' << LL(v_h[ind_h[blockSize / 2]]) << ' '
                << LL(v_h[ind_h[blockSize / 2 + 1]]) << std::endl;
    }
    for (int ib = 0; ib < blocks; ++ib) {
      std::set<uint16_t> inds;
      if (offsets_h[ib + 1] > offsets_h[ib])
        inds.insert(ind_h[offsets_h[ib]]);
      for (auto j = offsets_h[ib] + 1; j < offsets_h[ib + 1]; j++) {
        if (inds.count(ind_h[j]) != 0) {
          printf("i=%d ib=%d ind_h[j=%d]=%d: duplicate indice!\n", i, ib, j, ind_h[j]);
          std::vector<int> counts;
          counts.resize(offsets_h[ib + 1] - offsets_h[ib], 0);
          for (size_t j2 = offsets_h[ib]; j2 < offsets_h[ib + 1]; j2++) {
            counts[ind_h[j2]]++;
          }
          for (size_t j2 = 0; j2 < counts.size(); j2++) {
            if (counts[j2] != 1)
              printf("counts[%ld]=%d ", j2, counts[j2]);
          }
          printf("\n");
          printf("inds.count(ind_h[j] = %lu\n", inds.count(ind_h[j]));
        }
        assert(0 == inds.count(ind_h[j]));
        inds.insert(ind_h[j]);
        auto a = v_h.data() + offsets_h[ib];
        auto k1 = a[ind_h[j]];
        auto k2 = a[ind_h[j - 1]];
        truncate<NS>(k1);
        truncate<NS>(k2);
        if (k1 < k2) {
          std::cout << "i=" << i << " not ordered at ib=" << ib << " in [" << offsets_h[ib] << ", "
                    << offsets_h[ib + 1] - 1 << "] j=" << j << " ind[j]=" << ind_h[j]
                    << " (k1 < k2) : a1=" << (int64_t)a[ind_h[j]] << " k1=" << (int64_t)k1
                    << " a2= " << (int64_t)a[ind_h[j - 1]] << " k2=" << (int64_t)k2 << std::endl;
          //sleep(2);
          assert(false);
        }
      }
      if (!inds.empty()) {
        assert(0 == *inds.begin());
        assert(inds.size() - 1 == *inds.rbegin());
      }
      if (inds.size() != (offsets_h[ib + 1] - offsets_h[ib]))
        std::cout << "error " << i << ' ' << ib << ' ' << inds.size() << "!=" << (offsets_h[ib + 1] - offsets_h[ib])
                  << std::endl;
      //
      assert(inds.size() == (offsets_h[ib + 1] - offsets_h[ib]));
    }
  }  // 50 times
  std::cout << "Kernel computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 50.
            << " ms per pass" << std::endl;
}

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
              << ", the test will be skipped.\n";
    return 0;
  }

  for (auto const& device : devices) {
    Queue queue(device);
    bool useShared = false;

    std::cout << "using Global memory" << std::endl;

    go<int8_t>(queue, useShared);
    go<int16_t>(queue, useShared);
    go<int32_t>(queue, useShared);
    go<int32_t, 3>(queue, useShared);
    go<int64_t>(queue, useShared);
    go<float, 4, float, double>(queue, useShared);
    go<float, 2, float, double>(queue, useShared);

    go<uint8_t>(queue, useShared);
    go<uint16_t>(queue, useShared);
    go<uint32_t>(queue, useShared);
    // go<uint64_t>(v_h);

    useShared = true;

    std::cout << "using Shared memory" << std::endl;

    go<int8_t>(queue, useShared);
    go<int16_t>(queue, useShared);
    go<int32_t>(queue, useShared);
    go<int32_t, 3>(queue, useShared);
    go<int64_t>(queue, useShared);
    go<float, 4, float, double>(queue, useShared);
    go<float, 2, float, double>(queue, useShared);

    go<uint8_t>(queue, useShared);
    go<uint16_t>(queue, useShared);
    go<uint32_t>(queue, useShared);
    // go<uint64_t>(v_h);
  }
  return 0;
}
