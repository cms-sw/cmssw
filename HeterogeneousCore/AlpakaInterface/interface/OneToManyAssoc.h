#ifndef HeterogeneousCore_AlpakaInterface_interface_OneToManyAssoc_h
#define HeterogeneousCore_AlpakaInterface_interface_OneToManyAssoc_h

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FlexiStorage.h"

namespace {
  template <typename T>
  ALPAKA_FN_HOST_ACC typename std::make_signed<T>::type toSigned2(T v) {
    return static_cast<typename std::make_signed<T>::type>(v);
  }
}  // namespace
namespace cms {
  namespace alpakatools {

    template <typename Assoc>
    struct OneToManyAssocView {
      using Counter = typename Assoc::Counter;
      using index_type = typename Assoc::index_type;

      Assoc *assoc = nullptr;
      Counter *offStorage = nullptr;
      index_type *contentStorage = nullptr;
      int32_t offSize = -1;
      int32_t contentSize = -1;
    };

    // this MUST BE DONE in a single block (or in two kernels!)
    struct zeroAndInit {
      template <typename TAcc, typename Assoc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, OneToManyAssocView<Assoc> view) const {
        auto h = view.assoc;
        ALPAKA_ASSERT_OFFLOAD((1 == alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]));
        ALPAKA_ASSERT_OFFLOAD((0 == alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]));

        auto first = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        if (0 == first) {
          h->psws = 0;
          h->initStorage(view);
        }
        alpaka::syncBlockThreads(acc);
        // TODO use for_each_element_in_grid_strided (or similar)
        for (int i = first, nt = h->totOnes(); i < nt;
             i += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0]) {
          h->off[i] = 0;
        }
      }
    };

    template <typename TAcc, typename Assoc, typename TQueue>
    inline __attribute__((always_inline)) void launchZero(Assoc *h, TQueue &queue) {
      typename Assoc::View view = {h, nullptr, nullptr, -1, -1};
      launchZero<TAcc>(view, queue);
    }

    template <typename TAcc, typename Assoc, typename TQueue>
    inline __attribute__((always_inline)) void launchZero(OneToManyAssocView<Assoc> view, TQueue &queue) {
      if constexpr (Assoc::ctCapacity() < 0) {
        ALPAKA_ASSERT_OFFLOAD(view.contentStorage);
        ALPAKA_ASSERT_OFFLOAD(view.contentSize > 0);
      }
      if constexpr (Assoc::ctNOnes() < 0) {
        ALPAKA_ASSERT_OFFLOAD(view.offStorage);
        ALPAKA_ASSERT_OFFLOAD(view.offSize > 0);
      }
#if !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
      auto nthreads = 1024;
      auto nblocks = 1;  // MUST BE ONE as memory is initialize in thread 0 (alternative is two kernels);
      auto workDiv = cms::alpakatools::make_workdiv<TAcc>(nblocks, nthreads);
      alpaka::exec<TAcc>(queue, workDiv, zeroAndInit{}, view);
#else
      auto h = view.assoc;
      ALPAKA_ASSERT_OFFLOAD(h);
      h->initStorage(view);
      h->zero();
      h->psws = 0;
#endif
    }

    template <typename TAcc, typename Assoc, typename TQueue>
    inline __attribute__((always_inline)) void launchFinalize(Assoc *h, TQueue &queue) {
      typename Assoc::View view = {h, nullptr, nullptr, -1, -1};
      launchFinalize<TAcc>(view, queue);
    }

    template <typename TAcc, typename Assoc, typename TQueue>
    inline __attribute__((always_inline)) void launchFinalize(OneToManyAssocView<Assoc> view, TQueue &queue) {
      auto h = view.assoc;
      ALPAKA_ASSERT_OFFLOAD(h);
#if !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
      using Counter = typename Assoc::Counter;
      Counter *poff = (Counter *)((char *)(h) + offsetof(Assoc, off));
      auto nOnes = Assoc::ctNOnes();
      if constexpr (Assoc::ctNOnes() < 0) {
        ALPAKA_ASSERT_OFFLOAD(view.offStorage);
        ALPAKA_ASSERT_OFFLOAD(view.offSize > 0);
        nOnes = view.offSize;
        poff = view.offStorage;
      }
      ALPAKA_ASSERT_OFFLOAD(nOnes > 0);
      int32_t *ppsws = (int32_t *)((char *)(h) + offsetof(Assoc, psws));
      auto nthreads = 1024;
      auto nblocks = (nOnes + nthreads - 1) / nthreads;
      auto workDiv = cms::alpakatools::make_workdiv<TAcc>(nblocks, nthreads);
      alpaka::exec<TAcc>(queue,
                         workDiv,
                         multiBlockPrefixScan<Counter>(),
                         poff,
                         poff,
                         nOnes,
                         nblocks,
                         ppsws,
                         alpaka::getWarpSizes(alpaka::getDev(queue))[0]);
#else
      h->finalize();
#endif
    }

    struct finalizeBulk {
      template <typename TAcc, typename Assoc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, AtomicPairCounter const *apc, Assoc *__restrict__ assoc) const {
        assoc->bulkFinalizeFill(acc, *apc);
      }
    };

    template <typename I,    // type stored in the container (usually an index in a vector of the input values)
              int32_t ONES,  // number of "Ones"  +1. If -1 is initialized at runtime using external storage
              int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
              >
    class OneToManyAssoc {
    public:
      using View = OneToManyAssocView<OneToManyAssoc<I, ONES, SIZE>>;
      using Counter = uint32_t;

      using CountersOnly = OneToManyAssoc<I, ONES, 0>;

      using index_type = I;


      static constexpr int32_t ctNOnes() { return ONES; }
      constexpr auto totOnes() const { return off.capacity(); }
      constexpr auto nOnes() const { return totOnes() - 1; }
      static constexpr int32_t ctCapacity() { return SIZE; }
      constexpr auto capacity() const { return content.capacity(); }

      ALPAKA_FN_HOST_ACC void initStorage(View view) {
        ALPAKA_ASSERT_OFFLOAD(view.assoc == this);
        if constexpr (ctCapacity() < 0) {
          ALPAKA_ASSERT_OFFLOAD(view.contentStorage);
          ALPAKA_ASSERT_OFFLOAD(view.contentSize > 0);
          content.init(view.contentStorage, view.contentSize);
        }
        if constexpr (ctNOnes() < 0) {
          ALPAKA_ASSERT_OFFLOAD(view.offStorage);
          ALPAKA_ASSERT_OFFLOAD(view.offSize > 0);
          off.init(view.offStorage, view.offSize);
        }
      }

      ALPAKA_FN_HOST_ACC void zero() {
        for (int32_t i = 0; i < totOnes(); ++i) {
          off[i] = 0;
        }
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void add(const TAcc &acc, CountersOnly const &co) {
        for (uint32_t i = 0; toSigned2(i) < totOnes(); ++i) {
          alpaka::atomicAdd(acc, off.data() + i, co.off[i], alpaka::hierarchy::Blocks{});
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
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void count(const TAcc &acc, I b) {
        ALPAKA_ASSERT_OFFLOAD(b < static_cast<uint32_t>(nOnes()));
        atomicIncrement(acc, off[b]);
      }

      template <typename TAcc>
      ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const TAcc &acc, I b, index_type j) {
        ALPAKA_ASSERT_OFFLOAD(b < static_cast<uint32_t>(nOnes()));
        auto w = atomicDecrement(acc, off[b]);
        ALPAKA_ASSERT_OFFLOAD(w > 0);
        content[w - 1] = j;
      }

      template <typename TAcc>
      ALPAKA_FN_HOST_ACC inline __attribute__((always_inline)) int32_t bulkFill(const TAcc &acc,
                                                                                AtomicPairCounter &apc,
                                                                                index_type const *v,
                                                                                uint32_t n) {
        auto c = apc.inc_add(acc, n);
        if (int(c.first) >= nOnes())
          return -int32_t(c.first);
        off[c.first] = c.second;
        for (uint32_t j = 0; j < n; ++j)
          content[c.second + j] = v[j];
        return c.first;
      }

      ALPAKA_FN_HOST_ACC inline __attribute__((always_inline)) void bulkFinalize(AtomicPairCounter const &apc) {
        off[apc.get().first] = apc.get().second;
      }

      template <typename TAcc>
      ALPAKA_FN_HOST_ACC inline __attribute__((always_inline)) void bulkFinalizeFill(TAcc &acc,
                                                                                     AtomicPairCounter const &apc) {
        int f = apc.get().first;
        auto s = apc.get().second;
        if (f >= nOnes()) {  // overflow!
          off[nOnes()] = uint32_t(off[nOnes() - 1]);
          return;
        }
        auto first = f + alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        for (int i = first; i < totOnes(); i += alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0]) {
          off[i] = s;
        }
      }

      template <typename TAcc>
      ALPAKA_FN_HOST_ACC inline __attribute__((always_inline)) void finalize(TAcc &acc, Counter *ws = nullptr) {
        ALPAKA_ASSERT_OFFLOAD(off[totOnes() - 1] == 0);
        blockPrefixScan(acc, off.data(), totOnes(), ws);
        ALPAKA_ASSERT_OFFLOAD(off[totOnes() - 1] == off[totOnes() - 2]);
      }

      ALPAKA_FN_HOST_ACC inline __attribute__((always_inline)) void finalize() {
        // Single thread finalize.
        for (uint32_t i = 1; toSigned2(i) < totOnes(); ++i)
          off[i] += off[i - 1];
      }

      constexpr auto size() const {
        //printf ("In OneToManyAssoc::size(): totOnes()=%d, size = %d\n", totOnes(), off[totOnes() - 1]);
        return uint32_t(off[totOnes() - 1]);
      }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return content.data(); }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return content.data() + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return content.data() + off[b + 1]; }

      FlexiStorage<Counter, ONES> off;
      int32_t psws;  // prefix-scan working space
      FlexiStorage<index_type, SIZE> content;
    };

  }  // namespace alpakatools
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
