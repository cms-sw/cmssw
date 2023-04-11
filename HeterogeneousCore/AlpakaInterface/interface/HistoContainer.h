#ifndef HeterogeneousCore_AlpakaInterface_interface_HistoContainer_h
#define HeterogeneousCore_AlpakaInterface_interface_HistoContainer_h

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "HeterogeneousCore/AlpakaUtilities/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/alpakastdAlgorithm.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/prefixScan.h"

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
namespace cms {
  namespace alpakatools {

    struct countFromVector {
      template <typename TAcc, typename Histo, typename T>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    Histo *__restrict__ h,
                                    uint32_t nh,
                                    T const *__restrict__ v,
                                    uint32_t const *__restrict__ offsets) const {
        const uint32_t nt = offsets[nh];
        for_each_element_in_grid_strided(acc, nt, [&](uint32_t i) {
          auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
          ALPAKA_ASSERT_OFFLOAD((*off) > 0);
          int32_t ih = off - offsets - 1;
          ALPAKA_ASSERT_OFFLOAD(ih >= 0);
          ALPAKA_ASSERT_OFFLOAD(ih < int(nh));
          h->count(acc, v[i], ih);
        });
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
        for_each_element_in_grid_strided(acc, nt, [&](uint32_t i) {
          auto off = alpaka_std::upper_bound(offsets, offsets + nh + 1, i);
          ALPAKA_ASSERT_OFFLOAD((*off) > 0);
          int32_t ih = off - offsets - 1;
          ALPAKA_ASSERT_OFFLOAD(ih >= 0);
          ALPAKA_ASSERT_OFFLOAD(ih < int(nh));
          h->fill(acc, v[i], i, ih);
        });
      }
    };

    template <typename TAcc, typename Histo, typename TQueue>
    inline __attribute__((always_inline)) void launchZero(Histo *__restrict__ h, TQueue &queue) {
      auto histoOffView = make_device_view(alpaka::getDev(queue), h->off, Histo::totbins());
      alpaka::memset(queue, histoOffView, 0);
    }

    template <typename TAcc, typename Histo, typename TQueue>
    inline __attribute__((always_inline)) void launchFinalize(Histo *__restrict__ h, TQueue &queue) {
      uint32_t *poff = h->off;

      const int num_items = Histo::totbins();

      const auto threadsPerBlockOrElementsPerThread = 1024u;
      const auto blocksPerGrid = divide_up_by(num_items, threadsPerBlockOrElementsPerThread);
      const auto workDiv = make_workdiv<TAcc>(blocksPerGrid, threadsPerBlockOrElementsPerThread);
      alpaka::exec<TAcc>(queue, workDiv, multiBlockPrefixScanFirstStep<uint32_t>(), poff, poff, num_items);

      const auto workDivWith1Block = make_workdiv<TAcc>(1, threadsPerBlockOrElementsPerThread);
      alpaka::exec<TAcc>(
          queue, workDivWith1Block, multiBlockPrefixScanSecondStep<uint32_t>(), poff, poff, num_items, blocksPerGrid);
    }

    template <typename TAcc, typename Histo, typename T, typename TQueue>
    inline __attribute__((always_inline)) void fillManyFromVector(Histo *__restrict__ h,
                                                                  uint32_t nh,
                                                                  T const *v,
                                                                  uint32_t const *offsets,
                                                                  uint32_t totSize,
                                                                  uint32_t nthreads,
                                                                  TQueue &queue) {
      launchZero<TAcc>(h, queue);

      const auto threadsPerBlockOrElementsPerThread = nthreads;
      const auto blocksPerGrid = divide_up_by(totSize, nthreads);
      const auto workDiv = make_workdiv<TAcc>(blocksPerGrid, threadsPerBlockOrElementsPerThread);

      alpaka::exec<TAcc>(queue, workDiv, countFromVector(), h, nh, v, offsets);
      launchFinalize<TAcc>(h, queue);

      alpaka::exec<TAcc>(queue, workDiv, fillFromVector(), h, nh, v, offsets);
    }

    struct finalizeBulk {
      template <typename TAcc, typename Assoc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, AtomicPairCounter const *apc, Assoc *__restrict__ assoc) const {
        assoc->bulkFinalizeFill(acc, *apc);
      }
    };

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      ALPAKA_ASSERT_OFFLOAD(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      ALPAKA_ASSERT_OFFLOAD(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    template <typename T,                  // the type of the discretized input values
              uint32_t NBINS,              // number of bins //TODO: WTPM is going on here!?!
              int32_t SIZE,                // max number of element
              uint32_t S = sizeof(T) * 8,  // number of significant bits in T
              typename I = uint32_t,  // type stored in the container (usually an index in a vector of the input values)
              uint32_t NHISTS = 1     // number of histos stored
              >
    class HistoContainer {
    public:
      using Counter = uint32_t;

      using CountersOnly = HistoContainer<T, NBINS, 0, S, I, NHISTS>;

      using index_type = I;
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
      static constexpr uint32_t nbins() { return NBINS; }
      static constexpr int32_t nhists() { return NHISTS; }
      static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
      static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }
      static constexpr int32_t capacity() { return SIZE; }

      static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

      ALPAKA_FN_ACC ALPAKA_FN_INLINE void zero() {
        for (auto &i : off)
          i = 0;
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void add(const TAcc &acc, CountersOnly const &co) {
        for (uint32_t i = 0; i < totbins(); ++i) {
          alpaka::atomicAdd(acc, off + i, co.off[i], alpaka::hierarchy::Blocks{});
        }
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE static uint32_t atomicIncrement(const TAcc &acc, Counter &x) {
        return alpaka::atomicAdd(acc, &x, 1u, alpaka::hierarchy::Blocks{});
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE static uint32_t atomicDecrement(const TAcc &acc, Counter &x) {
        return alpaka::atomicSub(acc, &x, 1u, alpaka::hierarchy::Blocks{});
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void countDirect(const TAcc &acc, T b) {
        ALPAKA_ASSERT_OFFLOAD(b < nbins());
        atomicIncrement(acc, off[b]);
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillDirect(const TAcc &acc, T b, index_type j) {
        ALPAKA_ASSERT_OFFLOAD(b < nbins());
        auto w = atomicDecrement(acc, off[b]);
        ALPAKA_ASSERT_OFFLOAD(w > 0);
        bins[w - 1] = j;
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE int32_t
      bulkFill(const TAcc &acc, AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(acc, n);
        if (c.m >= nbins())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          bins[c.n + j] = v[j];
        return c.m;
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void bulkFinalize(const TAcc &acc, AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void bulkFinalizeFill(const TAcc &acc, AtomicPairCounter const &apc) {
        auto m = apc.get().m;
        auto n = apc.get().n;

        if (m >= nbins()) {  // overflow!
          off[nbins()] = uint32_t(off[nbins() - 1]);
          return;
        }

        for_each_element_in_grid_strided(acc, totbins(), m, [&](uint32_t i) { off[i] = n; });
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const TAcc &acc, T t) {
        uint32_t b = bin(t);
        ALPAKA_ASSERT_OFFLOAD(b < nbins());
        atomicIncrement(acc, off[b]);
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const TAcc &acc, T t, index_type j) {
        uint32_t b = bin(t);
        ALPAKA_ASSERT_OFFLOAD(b < nbins());
        auto w = atomicDecrement(acc, off[b]);
        ALPAKA_ASSERT_OFFLOAD(w > 0);
        bins[w - 1] = j;
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const TAcc &acc, T t, uint32_t nh) {
        uint32_t b = bin(t);
        ALPAKA_ASSERT_OFFLOAD(b < nbins());
        b += histOff(nh);
        ALPAKA_ASSERT_OFFLOAD(b < totbins());
        atomicIncrement(acc, off[b]);
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const TAcc &acc, T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        ALPAKA_ASSERT_OFFLOAD(b < nbins());
        b += histOff(nh);
        ALPAKA_ASSERT_OFFLOAD(b < totbins());
        auto w = atomicDecrement(acc, off[b]);
        ALPAKA_ASSERT_OFFLOAD(w > 0);
        bins[w - 1] = j;
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void finalize(const TAcc &acc, Counter *ws = nullptr) {
        ALPAKA_ASSERT_OFFLOAD(off[totbins() - 1] == 0);
        blockPrefixScan(acc, off, totbins(), ws);
        ALPAKA_ASSERT_OFFLOAD(off[totbins() - 1] == off[totbins() - 2]);
      }

      constexpr auto size() const { return uint32_t(off[totbins() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return bins; }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return bins + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return bins + off[b + 1]; }

      Counter off[totbins()];
      index_type bins[capacity()];
    };

    template <typename I,       // type stored in the container (usually an index in a vector of the input values)
              int32_t MAXONES,  // max number of "ones"
              int32_t MAXMANYS  // max number of "manys"
              >
    using OneToManyAssoc = HistoContainer<uint32_t, MAXONES, MAXMANYS, sizeof(uint32_t) * 8, I, 1>;

  }  // namespace alpakatools
}  // namespace cms
#endif  // HeterogeneousCore_AlpakaInterface_interface_HistoContainer_h
