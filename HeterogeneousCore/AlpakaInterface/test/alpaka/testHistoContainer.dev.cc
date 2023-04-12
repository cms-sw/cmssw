#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

using namespace cms::alpakatools;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;

static constexpr auto s_tag = "[" ALPAKA_TYPE_ALIAS_NAME(alpakaTestHistoContainer) "]";

template <typename T>
int go(const DevHost& host, const Device& device, Queue& queue) {
  std::mt19937 eng(2708);
  std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  constexpr unsigned int N = 12000;
  auto v = make_host_buffer<T[]>(queue, N);
  auto v_d = make_device_buffer<T[]>(queue, N);
  alpaka::memcpy(queue, v_d, v);

  constexpr uint32_t nParts = 10;
  constexpr uint32_t partSize = N / nParts;

  using Hist = HistoContainer<T, 128, N, 8 * sizeof(T), uint32_t, nParts>;
  std::cout << "HistoContainer " << (int)(offsetof(Hist, off)) << ' ' << Hist::nbins() << ' ' << Hist::totbins() << ' '
            << Hist::capacity() << ' ' << offsetof(Hist, bins) - offsetof(Hist, off) << ' '
            << (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / Hist::nbins() << std::endl;

  auto offsets = make_host_buffer<uint32_t[]>(queue, nParts + 1);
  auto offsets_d = make_device_buffer<uint32_t[]>(queue, nParts + 1);

  auto h = make_host_buffer<Hist>(queue);
  auto h_d = make_device_buffer<Hist>(queue);

  for (int it = 0; it < 5; ++it) {
    offsets[0] = 0;
    for (uint32_t j = 1; j < nParts + 1; ++j) {
      offsets[j] = offsets[j - 1] + partSize - 3 * j;
      assert(offsets[j] <= N);
    }

    if (it == 1) {  // special cases...
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

    alpaka::memcpy(queue, offsets_d, offsets);

    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);

    if (it == 2) {  // big bin
      for (long long j = 1000; j < 2000; j++)
        v[j] = sizeof(T) == 1 ? 22 : 3456;
    }

    // for(unsigned int i=0;i<N;i++)
    // {
    //   std::cout << "values>" << v[i] << std::endl;
    // }
    alpaka::memcpy(queue, v_d, v);

    alpaka::memset(queue, h_d, 0);

    alpaka::wait(queue);

    std::cout << "Calling fillManyFromVector - " << h->size() << std::endl;
    fillManyFromVector<Acc1D>(h_d.data(), nParts, v_d.data(), offsets_d.data(), offsets[10], 256, queue);

    alpaka::memcpy(queue, h, h_d);
    // std::cout << "Calling fillManyFromVector - " <<  h->size() << std::endl;
    alpaka::wait(queue);
    std::cout << "Copied results" << std::endl;
    // for(int i =0;i<=10;i++)
    // {
    //   std::cout << offsets[i] <<" - "<< h->size() << std::endl;
    // }

    assert(0 == h->off[0]);
    assert(offsets[10] == h->size());

    auto verify = [&](uint32_t i, uint32_t k, uint32_t t1, uint32_t t2) {
      assert(t1 < N);
      assert(t2 < N);
      if (T(v[t1] - v[t2]) <= 0)
        std::cout << "for " << i << ':' << v[k] << " failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };

    // make sure it spans 3 bins...
    auto window = T(1300);

    for (uint32_t j = 0; j < nParts; ++j) {
      auto off = Hist::histOff(j);
      for (uint32_t i = 0; i < Hist::nbins(); ++i) {
        auto ii = i + off;
        if (0 == h->size(ii))
          continue;
        auto k = *h->begin(ii);
        if (j % 2)
          k = *(h->begin(ii) + (h->end(ii) - h->begin(ii)) / 2);
#ifndef NDEBUG
        auto bk = h->bin(v[k]);
#endif
        assert(bk == i);
        assert(k < offsets[j + 1]);
        auto kl = h->bin(v[k] - window);
        auto kh = h->bin(v[k] + window);
        assert(kl != i);
        assert(kh != i);
        // std::cout << kl << ' ' << kh << std::endl;

        auto me = v[k];
        auto tot = 0;
        auto nm = 0;
        bool l = true;
        auto khh = kh;
        incr(khh);
        for (auto kk = kl; kk != khh; incr(kk)) {
          if (kk != kl && kk != kh)
            nm += h->size(kk + off);
          for (auto p = h->begin(kk + off); p < h->end(kk + off); ++p) {
            if (std::min(std::abs(T(v[*p] - me)), std::abs(T(me - v[*p]))) > window) {
            } else {
              ++tot;
            }
          }
          if (kk == i) {
            l = false;
            continue;
          }
          if (l)
            for (auto p = h->begin(kk + off); p < h->end(kk + off); ++p)
              verify(i, k, k, (*p));
          else
            for (auto p = h->begin(kk + off); p < h->end(kk + off); ++p)
              verify(i, k, (*p), k);
        }
        if (!(tot >= nm)) {
          std::cout << "too bad " << j << ' ' << i << ' ' << int(me) << '/' << (int)T(me - window) << '/'
                    << (int)T(me + window) << ": " << kl << '/' << kh << ' ' << khh << ' ' << tot << '/' << nm
                    << std::endl;
        }
        if (l)
          std::cout << "what? " << j << ' ' << i << ' ' << int(me) << '/' << (int)T(me - window) << '/'
                    << (int)T(me + window) << ": " << kl << '/' << kh << ' ' << khh << ' ' << tot << '/' << nm
                    << std::endl;
        assert(!l);
      }
    }
  }

  return 0;
}

TEST_CASE("Standard checks of " ALPAKA_TYPE_ALIAS_NAME(alpakaTestHistoContainer), s_tag) {
  SECTION("HistoContainerKernel") {
    // get the list of devices on the current platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    auto const& host = cms::alpakatools::host();
    if (devices.empty()) {
      std::cout << "No devices available on the platform " << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)
                << ", the test will be skipped.\n";
      return;
    }
    // run the test on each device
    for (auto const& device : devices) {
      std::cout << "Test Histo Container on " << alpaka::getName(device) << '\n';
      auto queue = Queue(device);

      REQUIRE(go<int16_t>(host, device, queue) == 0);
      REQUIRE(go<int8_t>(host, device, queue) == 0);
    }
  }
}
