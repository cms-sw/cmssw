#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainer_h

#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/maxCoopBlocks.h"
#endif

namespace cms {
  namespace cuda {

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
