#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <random>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

using namespace cms::cuda;

template <typename T>
void go() {
  std::mt19937 eng;
  std::uniform_int_distribution<T> rgen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  constexpr int N = 12000;
  T v[N];
  auto v_d = make_device_unique<T[]>(N, nullptr);

  cudaCheck(cudaMemcpy(v_d.get(), v, N * sizeof(T), cudaMemcpyHostToDevice));

  constexpr uint32_t nParts = 10;
  constexpr uint32_t partSize = N / nParts;
  uint32_t offsets[nParts + 1];

  using Hist = HistoContainer<T, 128, N, 8 * sizeof(T), uint32_t, nParts>;
  std::cout << "HistoContainer " << (int)(offsetof(Hist, off)) << ' ' << Hist::nbins() << ' ' << Hist::totbins() << ' '
            << Hist::ctCapacity() << ' ' << offsetof(Hist, content) - offsetof(Hist, off) << ' '
            << (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) / Hist::nbins() << std::endl;

  assert(Hist::totbins() == Hist::ctNOnes());

  Hist h;
  auto h_d = make_device_unique<Hist[]>(1, nullptr);

  auto off_d = make_device_unique<uint32_t[]>(nParts + 1, nullptr);

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
      assert(offsets[10] <= N);
    }

    cudaCheck(cudaMemcpy(off_d.get(), offsets, 4 * (nParts + 1), cudaMemcpyHostToDevice));

    for (long long j = 0; j < N; j++)
      v[j] = rgen(eng);

    if (it == 2) {  // big bin
      for (long long j = 1000; j < 2000; j++)
        v[j] = sizeof(T) == 1 ? 22 : 3456;
    }

    cudaCheck(cudaMemcpy(v_d.get(), v, N * sizeof(T), cudaMemcpyHostToDevice));

    fillManyFromVector(h_d.get(), nParts, v_d.get(), off_d.get(), offsets[10], 256, nullptr, 0);
    cudaCheck(cudaMemcpy(&h, h_d.get(), sizeof(Hist), cudaMemcpyDeviceToHost));
    assert(0 == h.off[0]);
    assert(offsets[10] == h.size());

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
        if (0 == h.size(ii))
          continue;
        auto k = *h.begin(ii);
        if (j % 2)
          k = *(h.begin(ii) + (h.end(ii) - h.begin(ii)) / 2);
        auto bk = h.bin(v[k]);
        assert(bk == i);
        assert(k < offsets[j + 1]);
        auto kl = h.bin(v[k] - window);
        auto kh = h.bin(v[k] + window);
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
            nm += h.size(kk + off);
          for (auto p = h.begin(kk + off); p < h.end(kk + off); ++p) {
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
            for (auto p = h.begin(kk + off); p < h.end(kk + off); ++p)
              verify(i, k, k, (*p));
          else
            for (auto p = h.begin(kk + off); p < h.end(kk + off); ++p)
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
}

int main() {
  cms::cudatest::requireDevices();

  go<int16_t>();
  go<int8_t>();

  return 0;
}
