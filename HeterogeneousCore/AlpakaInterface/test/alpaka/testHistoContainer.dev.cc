#include <algorithm>
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

template <typename T, typename Hist, typename VERIFY, typename INCR>
void checkContents(Hist* h,
                   unsigned int N,
                   VERIFY& verify,
                   INCR& incr,
                   T window,
                   uint32_t nParts,
                   const T* v,
                   const uint32_t* offsets) {
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
      [[maybe_unused]] auto bk = h->bin(v[k]);
#endif
      ALPAKA_ASSERT_ACC(bk == i);
      ALPAKA_ASSERT_ACC(k < offsets[j + 1]);
      auto kl = h->bin(v[k] - window);
      auto kh = h->bin(v[k] + window);
      ALPAKA_ASSERT_ACC(kl != i);
      ALPAKA_ASSERT_ACC(kh != i);
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
      ALPAKA_ASSERT_ACC(!l);
    }
  }
  int status;
  auto* demangled = abi::__cxa_demangle(typeid(Hist).name(), NULL, NULL, &status);
  status || printf("Check contents OK with %s\n", demangled);
  std::free(demangled);
}

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
  using HistR = HistoContainer<T, 128, -1, 8 * sizeof(T), uint32_t, nParts>;
  std::cout << "HistoContainer " << (int)(offsetof(Hist, off)) << ' ' << Hist::nbins() << ' ' << Hist::totbins() << ' '
            << Hist{}.capacity() << ' ' << offsetof(Hist, content) - offsetof(Hist, off) << ' '
            << (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / Hist::nbins() << std::endl;
  std::cout << "HistoContainer Runtime sized " << (int)(offsetof(HistR, off)) << ' ' << HistR::nbins() << ' '
            << HistR::totbins() << ' ' << HistR{}.capacity() << ' ' << offsetof(HistR, content) - offsetof(HistR, off)
            << ' ' << (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / HistR::nbins() << std::endl;

  // Offsets for each histogram.
  auto offsets = make_host_buffer<uint32_t[]>(queue, nParts + 1);
  auto offsets_d = make_device_buffer<uint32_t[]>(queue, nParts + 1);

  // Compile sized histogram (self contained)
  auto h = make_host_buffer<Hist>(queue);
  auto h_d = make_device_buffer<Hist>(queue);

  // Run time sized histogram
  auto hr = make_host_buffer<HistR>(queue);
  // Data storage for histogram content (host)
  auto hd = make_host_buffer<typename HistR::index_type[]>(queue, N);
  auto hr_d = make_device_buffer<HistR>(queue);
  // Data storage for histogram content (device)
  auto hd_d = make_device_buffer<typename HistR::index_type[]>(queue, N);

  // We iterate the test 5 times.
  for (int it = 0; it < 5; ++it) {
    offsets[0] = 0;
    for (uint32_t j = 1; j < nParts + 1; ++j) {
      offsets[j] = offsets[j - 1] + partSize - 3 * j;
      ALPAKA_ASSERT_ACC(offsets[j] <= N);
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
    alpaka::memset(queue, hr_d, 0);
    alpaka::memset(queue, hd_d, 0);

    alpaka::wait(queue);

    std::cout << "Calling fillManyFromVector - " << h->size() << std::endl;
    fillManyFromVector<Acc1D>(h_d.data(), nParts, v_d.data(), offsets_d.data(), offsets[10], 256, queue);

    std::cout << "Calling fillManyFromVector(runtime sized) - " << h->size() << std::endl;
    typename HistR::View hrv_d;
    hrv_d.assoc = hr_d.data();
    hrv_d.offSize = -1;
    hrv_d.offStorage = nullptr;
    hrv_d.contentSize = N;
    hrv_d.contentStorage = hd_d.data();
    fillManyFromVector<Acc1D>(hr_d.data(), hrv_d, nParts, v_d.data(), offsets_d.data(), offsets[10], 256, queue);

    alpaka::memcpy(queue, h, h_d);
    // For the runtime sized version:
    // We need the histogram for non external data (here, the offsets)
    // .. and external data storage (here, the contents)
    // .. and plug the data storage address into the histo container
    alpaka::memcpy(queue, hr, hr_d);
    alpaka::memcpy(queue, hd, hd_d);

    // std::cout << "Calling fillManyFromVector - " <<  h->size() << std::endl;
    alpaka::wait(queue);

    // We cannot update the contents address of the histo container before the copy from device happened
    typename HistR::View hrv;
    hrv.assoc = hr.data();
    hrv.offSize = -1;
    hrv.offStorage = nullptr;
    hrv.contentSize = N;
    hrv.contentStorage = hd.data();
    hr->initStorage(hrv);

    std::cout << "Copied results" << std::endl;
    // for(int i =0;i<=10;i++)
    // {
    //   std::cout << offsets[i] <<" - "<< h->size() << std::endl;
    // }

    ALPAKA_ASSERT_ACC(0 == h->off[0]);
    ALPAKA_ASSERT_ACC(offsets[10] == h->size());
    ALPAKA_ASSERT_ACC(0 == hr->off[0]);
    ALPAKA_ASSERT_ACC(offsets[10] == hr->size());

    auto verify = [&](uint32_t i, uint32_t k, uint32_t t1, uint32_t t2) {
      ALPAKA_ASSERT_ACC(t1 < N);
      ALPAKA_ASSERT_ACC(t2 < N);
      if (T(v[t1] - v[t2]) <= 0)
        std::cout << "for " << i << ':' << v[k] << " failed " << v[t1] << ' ' << v[t2] << std::endl;
    };

    auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };

    // make sure it spans 3 bins...
    auto window = T(1300);
    checkContents<T>(h.data(), N, verify, incr, window, nParts, v.data(), offsets.data());
    checkContents<T>(hr.data(), N, verify, incr, window, nParts, v.data(), offsets.data());
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
