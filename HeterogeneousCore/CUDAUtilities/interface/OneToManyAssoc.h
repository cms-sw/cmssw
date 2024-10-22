#ifndef HeterogeneousCore_CUDAUtilities_interface_OneToManyAssoc_h
#define HeterogeneousCore_CUDAUtilities_interface_OneToManyAssoc_h

#include <algorithm>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif  // __CUDA_ARCH__
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "HeterogeneousCore/CUDAUtilities/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"
#include "HeterogeneousCore/CUDAUtilities/interface/FlexiStorage.h"

namespace cms {
  namespace cuda {

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
    template <typename Assoc>
    __global__ void zeroAndInit(OneToManyAssocView<Assoc> view) {
      auto h = view.assoc;
      assert(1 == gridDim.x);
      assert(0 == blockIdx.x);

      int first = threadIdx.x;

      if (0 == first) {
        h->psws = 0;
        h->initStorage(view);
      }
      __syncthreads();
      for (int i = first, nt = h->totOnes(); i < nt; i += blockDim.x) {
        h->off[i] = 0;
      }
    }

    template <typename Assoc>
    inline __attribute__((always_inline)) void launchZero(Assoc *h,
                                                          cudaStream_t stream
#ifndef __CUDACC__
                                                          = cudaStreamDefault
#endif
    ) {
      typename Assoc::View view = {h, nullptr, nullptr, -1, -1};
      launchZero(view, stream);
    }
    template <typename Assoc>
    inline __attribute__((always_inline)) void launchZero(OneToManyAssocView<Assoc> view,
                                                          cudaStream_t stream
#ifndef __CUDACC__
                                                          = cudaStreamDefault
#endif
    ) {

      if constexpr (Assoc::ctCapacity() < 0) {
        assert(view.contentStorage);
        assert(view.contentSize > 0);
      }
      if constexpr (Assoc::ctNOnes() < 0) {
        assert(view.offStorage);
        assert(view.offSize > 0);
      }
#ifdef __CUDACC__
      auto nthreads = 1024;
      auto nblocks = 1;  // MUST BE ONE as memory is initialize in thread 0 (alternative is two kernels);
      zeroAndInit<<<nblocks, nthreads, 0, stream>>>(view);
      cudaCheck(cudaGetLastError());
#else
      auto h = view.assoc;
      assert(h);
      h->initStorage(view);
      h->zero();
      h->psws = 0;
#endif
    }

    template <typename Assoc>
    inline __attribute__((always_inline)) void launchFinalize(Assoc *h,
                                                              cudaStream_t stream
#ifndef __CUDACC__
                                                              = cudaStreamDefault
#endif
    ) {
      typename Assoc::View view = {h, nullptr, nullptr, -1, -1};
      launchFinalize(view, stream);
    }

    template <typename Assoc>
    inline __attribute__((always_inline)) void launchFinalize(OneToManyAssocView<Assoc> view,
                                                              cudaStream_t stream
#ifndef __CUDACC__
                                                              = cudaStreamDefault
#endif
    ) {
      auto h = view.assoc;
      assert(h);
#ifdef __CUDACC__
      using Counter = typename Assoc::Counter;
      Counter *poff = (Counter *)((char *)(h) + offsetof(Assoc, off));
      auto nOnes = Assoc::ctNOnes();
      if constexpr (Assoc::ctNOnes() < 0) {
        assert(view.offStorage);
        assert(view.offSize > 0);
        nOnes = view.offSize;
        poff = view.offStorage;
      }
      assert(nOnes > 0);
      int32_t *ppsws = (int32_t *)((char *)(h) + offsetof(Assoc, psws));
      auto nthreads = 1024;
      auto nblocks = (nOnes + nthreads - 1) / nthreads;
      multiBlockPrefixScan<<<nblocks, nthreads, sizeof(int32_t) * nblocks, stream>>>(poff, poff, nOnes, ppsws);
      cudaCheck(cudaGetLastError());
#else
      h->finalize();
#endif
    }

    template <typename Assoc>
    __global__ void finalizeBulk(AtomicPairCounter const *apc, Assoc *__restrict__ assoc) {
      assoc->bulkFinalizeFill(*apc);
    }

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

      static constexpr int32_t ctNOnes() { return ONES; }
      constexpr auto totOnes() const { return off.capacity(); }
      constexpr auto nOnes() const { return totOnes() - 1; }
      static constexpr int32_t ctCapacity() { return SIZE; }
      constexpr auto capacity() const { return content.capacity(); }

      __host__ __device__ void initStorage(View view) {
        assert(view.assoc == this);
        if constexpr (ctCapacity() < 0) {
          assert(view.contentStorage);
          assert(view.contentSize > 0);
          content.init(view.contentStorage, view.contentSize);
        }
        if constexpr (ctNOnes() < 0) {
          assert(view.offStorage);
          assert(view.offSize > 0);
          off.init(view.offStorage, view.offSize);
        }
      }

      __host__ __device__ void zero() {
        for (int32_t i = 0; i < totOnes(); ++i) {
          off[i] = 0;
        }
      }

      __host__ __device__ __forceinline__ void add(CountersOnly const &co) {
        for (int32_t i = 0; i < totOnes(); ++i) {
#ifdef __CUDA_ARCH__
          atomicAdd(off.data() + i, co.off[i]);
#else
          auto &a = (std::atomic<Counter> &)(off[i]);
          a += co.off[i];
#endif
        }
      }

      static __host__ __device__ __forceinline__ uint32_t atomicIncrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicAdd(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a++;
#endif
      }

      static __host__ __device__ __forceinline__ uint32_t atomicDecrement(Counter &x) {
#ifdef __CUDA_ARCH__
        return atomicSub(&x, 1);
#else
        auto &a = (std::atomic<Counter> &)(x);
        return a--;
#endif
      }

      __host__ __device__ __forceinline__ void count(int32_t b) {
        assert(b < nOnes());
        atomicIncrement(off[b]);
      }

      __host__ __device__ __forceinline__ void fill(int32_t b, index_type j) {
        assert(b < nOnes());
        auto w = atomicDecrement(off[b]);
        assert(w > 0);
        content[w - 1] = j;
      }

      __host__ __device__ __forceinline__ int32_t bulkFill(AtomicPairCounter &apc, index_type const *v, uint32_t n) {
        auto c = apc.add(n);
        if (int(c.m) >= nOnes())
          return -int32_t(c.m);
        off[c.m] = c.n;
        for (uint32_t j = 0; j < n; ++j)
          content[c.n + j] = v[j];
        return c.m;
      }

      __host__ __device__ __forceinline__ void bulkFinalize(AtomicPairCounter const &apc) {
        off[apc.get().m] = apc.get().n;
      }

      __host__ __device__ __forceinline__ void bulkFinalizeFill(AtomicPairCounter const &apc) {
        int m = apc.get().m;
        auto n = apc.get().n;
        if (m >= nOnes()) {  // overflow!
          off[nOnes()] = uint32_t(off[nOnes() - 1]);
          return;
        }
        auto first = m + blockDim.x * blockIdx.x + threadIdx.x;
        for (int i = first; i < totOnes(); i += gridDim.x * blockDim.x) {
          off[i] = n;
        }
      }

      __host__ __device__ __forceinline__ void finalize(Counter *ws = nullptr) {
        assert(off[totOnes() - 1] == 0);
        blockPrefixScan(off.data(), totOnes(), ws);
        assert(off[totOnes() - 1] == off[totOnes() - 2]);
      }

      constexpr auto size() const { return uint32_t(off[totOnes() - 1]); }
      constexpr auto size(uint32_t b) const { return off[b + 1] - off[b]; }

      constexpr index_type const *begin() const { return content.data(); }
      constexpr index_type const *end() const { return begin() + size(); }

      constexpr index_type const *begin(uint32_t b) const { return content.data() + off[b]; }
      constexpr index_type const *end(uint32_t b) const { return content.data() + off[b + 1]; }

      FlexiStorage<Counter, ONES> off;
      int32_t psws;  // prefix-scan working space
      FlexiStorage<index_type, SIZE> content;
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
