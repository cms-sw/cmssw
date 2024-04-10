#ifndef HeterogeneousCore_AlpakaInterface_interface_HistoContainer_h
#define HeterogeneousCore_AlpakaInterface_interface_HistoContainer_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/AlpakaInterface/interface/alpakastdAlgorithm.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace cms::alpakatools {

  struct countFromVector {
    template <typename TAcc, typename Histo, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  Histo *__restrict__ h,
                                  uint32_t nh,
                                  T const *__restrict__ v,
                                  uint32_t const *__restrict__ offsets) const {
      const uint32_t nt = offsets[nh];
      for (uint32_t i : uniform_elements(acc, nt)) {
        auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
        ALPAKA_ASSERT_ACC((*off) > 0);
        int32_t ih = off - offsets - 1;
        ALPAKA_ASSERT_ACC(ih >= 0);
        ALPAKA_ASSERT_ACC(ih < int(nh));
        h->count(acc, v[i], ih);
      }
    }
  };

  struct fillFromVector {
    template <typename TAcc, typename Histo, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  Histo *__restrict__ h,
                                  uint32_t nh,
                                  T const *__restrict__ v,
                                  uint32_t const *__restrict__ offsets) const {
      const uint32_t nt = offsets[nh];
      for (uint32_t i : uniform_elements(acc, nt)) {
        auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
        ALPAKA_ASSERT_ACC((*off) > 0);
        int32_t ih = off - offsets - 1;
        ALPAKA_ASSERT_ACC(ih >= 0);
        ALPAKA_ASSERT_ACC(ih < int(nh));
        h->fill(acc, v[i], i, ih);
      }
    }
  };

  template <typename TAcc, typename Histo, typename T, typename TQueue>
  ALPAKA_FN_INLINE void fillManyFromVector(Histo *__restrict__ h,
                                           uint32_t nh,
                                           T const *__restrict__ v,
                                           uint32_t const *__restrict__ offsets,
                                           uint32_t totSize,
                                           uint32_t nthreads,
                                           TQueue &queue) {
    Histo::template launchZero<TAcc>(h, queue);

    const auto threadsPerBlockOrElementsPerThread = nthreads;
    const auto blocksPerGrid = divide_up_by(totSize, nthreads);
    const auto workDiv = make_workdiv<TAcc>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

    alpaka::exec<TAcc>(queue, workDiv, countFromVector(), h, nh, v, offsets);
    Histo::template launchFinalize<TAcc>(h, queue);

    alpaka::exec<TAcc>(queue, workDiv, fillFromVector(), h, nh, v, offsets);
  }

  template <typename TAcc, typename Histo, typename T, typename TQueue>
  ALPAKA_FN_INLINE void fillManyFromVector(Histo *__restrict__ h,
                                           typename Histo::View hv,
                                           uint32_t nh,
                                           T const *__restrict__ v,
                                           uint32_t const *__restrict__ offsets,
                                           uint32_t totSize,
                                           uint32_t nthreads,
                                           TQueue &queue) {
    Histo::template launchZero<TAcc>(hv, queue);

    const auto threadsPerBlockOrElementsPerThread = nthreads;
    const auto blocksPerGrid = divide_up_by(totSize, nthreads);
    const auto workDiv = make_workdiv<TAcc>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

    alpaka::exec<TAcc>(queue, workDiv, countFromVector(), h, nh, v, offsets);
    Histo::template launchFinalize<TAcc>(h, queue);

    alpaka::exec<TAcc>(queue, workDiv, fillFromVector(), h, nh, v, offsets);
  }

  // iteratate over N bins left and right of the one containing "v"
  template <typename Hist, typename V, typename Func>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachInBins(Hist const &hist, V value, int n, Func func) {
    int bs = Hist::bin(value);
    int be = std::min(int(Hist::nbins() - 1), bs + n);
    bs = std::max(0, bs - n);
    ALPAKA_ASSERT_ACC(be >= bs);
    for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
      func(*pj);
    }
  }

  // iteratate over bins containing all values in window wmin, wmax
  template <typename Hist, typename V, typename Func>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
    auto bs = Hist::bin(wmin);
    auto be = Hist::bin(wmax);
    ALPAKA_ASSERT_ACC(be >= bs);
    for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
      func(*pj);
    }
  }

  template <typename T,      // the type of the discretized input values
            uint32_t NBINS,  // number of bins
            int32_t SIZE,    // max number of element. If -1 is initialized at runtime using external storage
            uint32_t S = sizeof(T) * 8,  // number of significant bits in T
            typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
            uint32_t NHISTS = 1     // number of histos stored
            >
  class HistoContainer : public OneToManyAssocRandomAccess<I, NHISTS * NBINS + 1, SIZE> {
  public:
    using Base = OneToManyAssocRandomAccess<I, NHISTS * NBINS + 1, SIZE>;
    using View = typename Base::View;
    using Counter = typename Base::Counter;
    using index_type = typename Base::index_type;
    using UT = typename std::make_unsigned<T>::type;

    static constexpr uint32_t ilog2(uint32_t v) {
      constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
      constexpr uint32_t s[] = {1, 2, 4, 8, 16};

      uint32_t r = 0;  // result of log2(v) will go here
      for (auto i = 4; i >= 0; i--)
        if (v & b[i]) {
          v >>= s[i];
          r |= s[i];
        }
      return r;
    }

    static constexpr uint32_t sizeT() { return S; }
    static constexpr int32_t nhists() { return NHISTS; }
    static constexpr uint32_t nbins() { return NBINS; }
    static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
    static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }

    static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

    static constexpr UT bin(T t) {
      constexpr uint32_t shift = sizeT() - nbits();
      constexpr uint32_t mask = (1 << nbits()) - 1;
      return (t >> shift) & mask;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const TAcc &acc, T t) {
      uint32_t b = bin(t);
      ALPAKA_ASSERT_ACC(b < nbins());
      Base::atomicIncrement(acc, this->off[b]);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const TAcc &acc, T t, index_type j) {
      uint32_t b = bin(t);
      ALPAKA_ASSERT_ACC(b < nbins());
      auto w = Base::atomicDecrement(acc, this->off[b]);
      ALPAKA_ASSERT_ACC(w > 0);
      this->content[w - 1] = j;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const TAcc &acc, T t, uint32_t nh) {
      uint32_t b = bin(t);
      ALPAKA_ASSERT_ACC(b < nbins());
      b += histOff(nh);
      ALPAKA_ASSERT_ACC(b < totbins());
      Base::atomicIncrement(acc, this->off[b]);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const TAcc &acc, T t, index_type j, uint32_t nh) {
      uint32_t b = bin(t);
      ALPAKA_ASSERT_ACC(b < nbins());
      b += histOff(nh);
      ALPAKA_ASSERT_ACC(b < totbins());
      auto w = Base::atomicDecrement(acc, this->off[b]);
      ALPAKA_ASSERT_ACC(w > 0);
      this->content[w - 1] = j;
    }
  };
}  // namespace cms::alpakatools
#endif  // HeterogeneousCore_AlpakaInterface_interface_HistoContainer_h
