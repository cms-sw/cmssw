#ifndef RecoTracker_LSTCore_interface_Common_h
#define RecoTracker_LSTCore_interface_Common_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/Common/interface/StdArray.h"

#if defined(FP16_Base)
#if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_fp16.h>
#elif defined ALPAKA_ACC_GPU_HIP_ENABLED
#include <hip/hip_fp16.h>
#endif
#endif

namespace lst {

  // Named constants for pixelTypes
  enum PixelType : int8_t { kInvalid = -1, kHighPt = 0, kLowPtPosCurv = 1, kLowPtNegCurv = 2 };

  // Named types for LST objects
  enum LSTObjType { T5 = 4, pT3 = 5, pT5 = 7, pLS = 8 };

// If a compile time flag does not define PT_CUT, default to 0.8 (GeV)
#ifndef PT_CUT
  constexpr float PT_CUT = 0.8f;
#endif

  constexpr unsigned int max_blocks = 80;
  constexpr unsigned int max_connected_modules = 40;

  constexpr unsigned int n_max_pixel_segments_per_module = 500000;

  constexpr unsigned int n_max_pixel_md_per_modules = 2 * n_max_pixel_segments_per_module;

  constexpr unsigned int n_max_pixel_triplets = 5000;
  constexpr unsigned int n_max_pixel_quintuplets = 15000;

  constexpr unsigned int n_max_pixel_track_candidates = 300000;
  constexpr unsigned int n_max_nonpixel_track_candidates = 10000;

  constexpr unsigned int size_superbins = 45000;

// Half precision wrapper functions.
#if defined(FP16_Base)
#define __F2H __float2half
#define __H2F __half2float
  typedef __half FPX;
#else
#define __F2H
#define __H2F
  typedef float FPX;
#endif

// Needed for files that are compiled by g++ to not throw an error.
// uint4 is defined only for CUDA, so we will have to revisit this soon when running on other backends.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
  };
#endif

  // Defining the constant host device variables right up here
  // Currently pixel tracks treated as LSs with 2 double layers (IT layers 1+2 and 3+4) and 4 hits. To be potentially handled better in the future.
  struct Params_Modules {
    using ArrayU16xMaxConnected = edm::StdArray<uint16_t, max_connected_modules>;
  };
  struct Params_pLS {
    static constexpr int kLayers = 2, kHits = 4;
  };
  struct Params_LS {
    static constexpr int kLayers = 2, kHits = 4;
    using ArrayUxLayers = edm::StdArray<unsigned int, kLayers>;
  };
  struct Params_T3 {
    static constexpr int kLayers = 3, kHits = 6;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
  };
  struct Params_pT3 {
    static constexpr int kLayers = 5, kHits = 10;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
  };
  struct Params_T5 {
    static constexpr int kLayers = 5, kHits = 10;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
  };
  struct Params_pT5 {
    static constexpr int kLayers = 7, kHits = 14;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
  };

  using ArrayIx2 = edm::StdArray<int, 2>;
  using ArrayUx2 = edm::StdArray<unsigned int, 2>;

}  //namespace lst

#endif
