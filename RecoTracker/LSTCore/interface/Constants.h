#ifndef RecoTracker_LSTCore_interface_Constants_h
#define RecoTracker_LSTCore_interface_Constants_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#ifdef CACHE_ALLOC
#include "HeterogeneousCore/AlpakaInterface/interface/CachedBufAlloc.h"
#endif

namespace lst {

  // Buffer type for allocations where auto type can't be used.
  template <typename TDev, typename TData>
  using Buf = alpaka::Buf<TDev, TData, alpaka_common::Dim1D, alpaka_common::Idx>;

  // Allocation wrapper function to make integration of the caching allocator easier and reduce code boilerplate.
  template <typename T, typename TDev, typename TSize, typename TQueue>
  ALPAKA_FN_HOST ALPAKA_FN_INLINE Buf<TDev, T> allocBufWrapper(TDev const& dev, TSize nElements, TQueue queue) {
#ifdef CACHE_ALLOC
    return cms::alpakatools::allocCachedBuf<T, alpaka_common::Idx>(
        dev, queue, alpaka_common::Vec1D(static_cast<alpaka_common::Idx>(nElements)));
#else
    return alpaka::allocBuf<T, alpaka_common::Idx>(dev,
                                                   alpaka_common::Vec1D(static_cast<alpaka_common::Idx>(nElements)));
#endif
  }

  // Second allocation wrapper function when queue is not given. Reduces code boilerplate.
  template <typename T, typename TDev, typename TSize>
  ALPAKA_FN_HOST ALPAKA_FN_INLINE Buf<TDev, T> allocBufWrapper(TDev const& dev, TSize nElements) {
    return alpaka::allocBuf<T, alpaka_common::Idx>(dev,
                                                   alpaka_common::Vec1D(static_cast<alpaka_common::Idx>(nElements)));
  }

  // Named constants for pixelTypes
  enum PixelType : int8_t { kInvalid = -1, kHighPt = 0, kLowPtPosCurv = 1, kLowPtNegCurv = 2 };

// If a compile time flag does not define PT_CUT, default to 0.8 (GeV)
#ifndef PT_CUT
  constexpr float PT_CUT = 0.8f;
#endif

  constexpr unsigned int max_blocks = 80;
  constexpr unsigned int max_connected_modules = 40;

  constexpr unsigned int n_max_pixel_segments_per_module = 50000;

  constexpr unsigned int n_max_pixel_md_per_modules = 2 * n_max_pixel_segments_per_module;

  constexpr unsigned int n_max_pixel_triplets = 5000;
  constexpr unsigned int n_max_pixel_quintuplets = 15000;

  constexpr unsigned int n_max_pixel_track_candidates = 30000;
  constexpr unsigned int n_max_nonpixel_track_candidates = 1000;

  constexpr unsigned int size_superbins = 45000;

  // Defining the constant host device variables right up here
  // Currently pixel tracks treated as LSs with 2 double layers (IT layers 1+2 and 3+4) and 4 hits. To be potentially handled better in the future.
  struct Params_pLS {
    static constexpr int kLayers = 2, kHits = 4;
  };
  struct Params_LS {
    static constexpr int kLayers = 2, kHits = 4;
  };
  struct Params_T3 {
    static constexpr int kLayers = 3, kHits = 6;
  };
  struct Params_pT3 {
    static constexpr int kLayers = 5, kHits = 10;
  };
  struct Params_T5 {
    static constexpr int kLayers = 5, kHits = 10;
  };
  struct Params_pT5 {
    static constexpr int kLayers = 7, kHits = 14;
  };

}  //namespace lst

#endif
