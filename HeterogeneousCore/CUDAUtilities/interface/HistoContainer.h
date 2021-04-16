#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"

namespace cms {
  namespace cuda {

    template <typename Histo, typename T>
    __global__ void countFromVector(Histo *__restrict__ h,
                                    uint32_t nh,
                                    T const *__restrict__ v,
                                    uint32_t const *__restrict__ offsets) {
      int first = blockDim.x * blockIdx.x + threadIdx.x;
      for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).count(v[i], ih);
      }
    }

    template <typename Histo, typename T>
    __global__ void fillFromVector(Histo *__restrict__ h,
                                   uint32_t nh,
                                   T const *__restrict__ v,
                                   uint32_t const *__restrict__ offsets) {
      int first = blockDim.x * blockIdx.x + threadIdx.x;
      for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        (*h).fill(v[i], i, ih);
      }
    }

    template <typename Histo, typename T>
    inline __attribute__((always_inline)) void fillManyFromVector(Histo *__restrict__ h,
                                                                  uint32_t nh,
                                                                  T const *__restrict__ v,
                                                                  uint32_t const *__restrict__ offsets,
                                                                  int32_t totSize,
                                                                  int nthreads,
                                                                  typename Histo::index_type *mem,
                                                                  cudaStream_t stream
#ifndef __CUDACC__
                                                                  = cudaStreamDefault
#endif
    ) {
      typename Histo::View view = {h, nullptr, mem, -1, totSize};
      launchZero(view, stream);
#ifdef __CUDACC__
      auto nblocks = (totSize + nthreads - 1) / nthreads;
      assert(nblocks > 0);
      countFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
      launchFinalize(view, stream);
      fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
#else
      countFromVector(h, nh, v, offsets);
      h->finalize();
      fillFromVector(h, nh, v, offsets);
#endif
    }

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    __host__ __device__ __forceinline__ void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
    __host__ __device__ __forceinline__ void forEachInWindow(Hist const &hist, V wmin, V wmax, Func const &func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      assert(be >= bs);
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
    class HistoContainer : public OneToManyAssoc<I, NHISTS * NBINS + 1, SIZE> {
    public:
      using Base = OneToManyAssoc<I, NHISTS * NBINS + 1, SIZE>;
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
      static constexpr uint32_t nbins() { return NBINS; }
      static constexpr uint32_t nhists() { return NHISTS; }
      static constexpr uint32_t totbins() { return NHISTS * NBINS + 1; }
      static constexpr uint32_t nbits() { return ilog2(NBINS - 1) + 1; }

      // static_assert(int32_t(totbins())==Base::ctNOnes());

      static constexpr auto histOff(uint32_t nh) { return NBINS * nh; }

      static constexpr UT bin(T t) {
        constexpr uint32_t shift = sizeT() - nbits();
        constexpr uint32_t mask = (1 << nbits()) - 1;
        return (t >> shift) & mask;
      }

      __host__ __device__ __forceinline__ void count(T t) {
        uint32_t b = bin(t);
        assert(b < nbins());
        Base::atomicIncrement(this->off[b]);
      }

      __host__ __device__ __forceinline__ void fill(T t, index_type j) {
        uint32_t b = bin(t);
        assert(b < nbins());
        auto w = Base::atomicDecrement(this->off[b]);
        assert(w > 0);
        this->content[w - 1] = j;
      }

      __host__ __device__ __forceinline__ void count(T t, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        Base::atomicIncrement(this->off[b]);
      }

      __host__ __device__ __forceinline__ void fill(T t, index_type j, uint32_t nh) {
        uint32_t b = bin(t);
        assert(b < nbins());
        b += histOff(nh);
        assert(b < totbins());
        auto w = Base::atomicDecrement(this->off[b]);
        assert(w > 0);
        this->content[w - 1] = j;
      }
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
