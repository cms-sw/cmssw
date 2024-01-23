#ifndef HeterogeneousCore_AlpakaInterface_interface_OneToManyAssoc_h
#define HeterogeneousCore_AlpakaInterface_interface_OneToManyAssoc_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/FlexiStorage.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace cms::alpakatools {

  template <typename I,    // type stored in the container (usually an index in a vector of the input values)
            int32_t ONES,  // number of "Ones"  +1. If -1 is initialized at runtime using external storage
            int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
            >
  class OneToManyAssocBase {
  public:
    using Counter = uint32_t;

    using CountersOnly = OneToManyAssocBase<I, ONES, 0>;

    using index_type = I;

    struct View {
      OneToManyAssocBase *assoc = nullptr;
      Counter *offStorage = nullptr;
      index_type *contentStorage = nullptr;
      int32_t offSize = -1;
      int32_t contentSize = -1;
    };

    static constexpr int32_t ctNOnes() { return ONES; }
    constexpr auto totOnes() const { return off.capacity(); }
    constexpr auto nOnes() const { return totOnes() - 1; }
    static constexpr int32_t ctCapacity() { return SIZE; }
    constexpr auto capacity() const { return content.capacity(); }

    ALPAKA_FN_HOST_ACC void initStorage(View view) {
      ALPAKA_ASSERT_ACC(view.assoc == this);
      if constexpr (ctCapacity() < 0) {
        ALPAKA_ASSERT_ACC(view.contentStorage);
        ALPAKA_ASSERT_ACC(view.contentSize > 0);
        content.init(view.contentStorage, view.contentSize);
      }
      if constexpr (ctNOnes() < 0) {
        ALPAKA_ASSERT_ACC(view.offStorage);
        ALPAKA_ASSERT_ACC(view.offSize > 0);
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
      for (uint32_t i = 0; static_cast<int>(i) < totOnes(); ++i) {
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
      ALPAKA_ASSERT_ACC(b < static_cast<uint32_t>(nOnes()));
      atomicIncrement(acc, off[b]);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fill(const TAcc &acc, I b, index_type j) {
      ALPAKA_ASSERT_ACC(b < static_cast<uint32_t>(nOnes()));
      auto w = atomicDecrement(acc, off[b]);
      ALPAKA_ASSERT_ACC(w > 0);
      content[w - 1] = j;
    }

    // this MUST BE DONE in a single block (or in two kernels!)
    struct zeroAndInit {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, View view) const {
        ALPAKA_ASSERT_ACC((1 == alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]));
        ALPAKA_ASSERT_ACC((0 == alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]));
        auto h = view.assoc;
        if (cms::alpakatools::once_per_block(acc)) {
          h->psws = 0;
          h->initStorage(view);
        }
        alpaka::syncBlockThreads(acc);
        for (int i : cms::alpakatools::independent_group_elements(acc, h->totOnes())) {
          h->off[i] = 0;
        }
      }
    };

    template <typename TAcc, typename TQueue>
    ALPAKA_FN_INLINE static void launchZero(OneToManyAssocBase *h, TQueue &queue) {
      View view = {h, nullptr, nullptr, -1, -1};
      launchZero<TAcc>(view, queue);
    }

    template <typename TAcc, typename TQueue>
    ALPAKA_FN_INLINE static void launchZero(View view, TQueue &queue) {
      if constexpr (ctCapacity() < 0) {
        ALPAKA_ASSERT_ACC(view.contentStorage);
        ALPAKA_ASSERT_ACC(view.contentSize > 0);
      }
      if constexpr (ctNOnes() < 0) {
        ALPAKA_ASSERT_ACC(view.offStorage);
        ALPAKA_ASSERT_ACC(view.offSize > 0);
      }
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        auto nthreads = 1024;
        auto nblocks = 1;  // MUST BE ONE as memory is initialize in thread 0 (alternative is two kernels);
        auto workDiv = cms::alpakatools::make_workdiv<TAcc>(nblocks, nthreads);
        alpaka::exec<TAcc>(queue, workDiv, zeroAndInit{}, view);
      } else {
        auto h = view.assoc;
        ALPAKA_ASSERT_ACC(h);
        h->initStorage(view);
        h->zero();
        h->psws = 0;
      }
    }

    constexpr auto size() const { return uint32_t(off[totOnes() - 1]); }
    constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

    constexpr index_type const *begin() const { return content.data(); }
    constexpr index_type const *end() const { return begin() + size(); }

    constexpr index_type const *begin(uint32_t b) const { return content.data() + off[b]; }
    constexpr index_type const *end(uint32_t b) const { return content.data() + off[b + 1]; }

    FlexiStorage<Counter, ONES> off;
    FlexiStorage<index_type, SIZE> content;
    int32_t psws;  // prefix-scan working space
  };

  template <typename I,    // type stored in the container (usually an index in a vector of the input values)
            int32_t ONES,  // number of "Ones"  +1. If -1 is initialized at runtime using external storage
            int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
            >
  class OneToManyAssocSequential : public OneToManyAssocBase<I, ONES, SIZE> {
  public:
    using index_type = typename OneToManyAssocBase<I, ONES, SIZE>::index_type;

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int32_t
    bulkFill(const TAcc &acc, AtomicPairCounter &apc, index_type const *v, uint32_t n) {
      auto c = apc.inc_add(acc, n);
      if (int(c.first) >= this->nOnes())
        return -int32_t(c.first);
      this->off[c.first] = c.second;
      for (uint32_t j = 0; j < n; ++j)
        this->content[c.second + j] = v[j];
      return c.first;
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void bulkFinalize(AtomicPairCounter const &apc) {
      this->off[apc.get().first] = apc.get().second;
    }

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void bulkFinalizeFill(TAcc &acc, AtomicPairCounter const &apc) {
      int f = apc.get().first;
      auto s = apc.get().second;
      if (f >= this->nOnes()) {  // overflow!
        this->off[this->nOnes()] = uint32_t(this->off[this->nOnes() - 1]);
        return;
      }
      auto first = f + alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      for (int i = first; i < this->totOnes(); i += alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0]) {
        this->off[i] = s;
      }
    }

    struct finalizeBulk {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    AtomicPairCounter const *apc,
                                    OneToManyAssocSequential *__restrict__ assoc) const {
        assoc->bulkFinalizeFill(acc, *apc);
      }
    };
  };

  template <typename I,    // type stored in the container (usually an index in a vector of the input values)
            int32_t ONES,  // number of "Ones"  +1. If -1 is initialized at runtime using external storage
            int32_t SIZE   // max number of element. If -1 is initialized at runtime using external storage
            >
  class OneToManyAssocRandomAccess : public OneToManyAssocBase<I, ONES, SIZE> {
  public:
    using Counter = typename OneToManyAssocBase<I, ONES, SIZE>::Counter;
    using View = typename OneToManyAssocBase<I, ONES, SIZE>::View;

    template <typename TAcc>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void finalize(TAcc &acc, Counter *ws = nullptr) {
      ALPAKA_ASSERT_ACC(this->off[this->totOnes() - 1] == 0);
      blockPrefixScan(acc, this->off.data(), this->totOnes(), ws);
      ALPAKA_ASSERT_ACC(this->off[this->totOnes() - 1] == this->off[this->totOnes() - 2]);
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void finalize() {
      // Single thread finalize.
      for (uint32_t i = 1; static_cast<int>(i) < this->totOnes(); ++i)
        this->off[i] += this->off[i - 1];
    }

    template <typename TAcc, typename TQueue>
    ALPAKA_FN_INLINE static void launchFinalize(OneToManyAssocRandomAccess *h, TQueue &queue) {
      View view = {h, nullptr, nullptr, -1, -1};
      launchFinalize<TAcc>(view, queue);
    }

    template <typename TAcc, typename TQueue>
    ALPAKA_FN_INLINE static void launchFinalize(View view, TQueue &queue) {
      // View stores a base pointer, we need to upcast back...
      auto h = static_cast<OneToManyAssocRandomAccess *>(view.assoc);
      ALPAKA_ASSERT_ACC(h);
      if constexpr (!requires_single_thread_per_block_v<TAcc>) {
        Counter *poff = (Counter *)((char *)(h) + offsetof(OneToManyAssocRandomAccess, off));
        auto nOnes = OneToManyAssocRandomAccess::ctNOnes();
        if constexpr (OneToManyAssocRandomAccess::ctNOnes() < 0) {
          ALPAKA_ASSERT_ACC(view.offStorage);
          ALPAKA_ASSERT_ACC(view.offSize > 0);
          nOnes = view.offSize;
          poff = view.offStorage;
        }
        ALPAKA_ASSERT_ACC(nOnes > 0);
        int32_t *ppsws = (int32_t *)((char *)(h) + offsetof(OneToManyAssocRandomAccess, psws));
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
      } else {
        h->finalize();
      }
    }
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
