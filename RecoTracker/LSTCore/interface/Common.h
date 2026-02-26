#ifndef RecoTracker_LSTCore_interface_Common_h
#define RecoTracker_LSTCore_interface_Common_h

#include "DataFormats/Common/interface/StdArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

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
  enum LSTObjType : int8_t { T5 = 4, pT3 = 5, pT5 = 7, pLS = 8, T4 = 9 };

  enum class HitType : int { Pixel = 0, Invalid = 3, Phase2OT = 4 };  // as in TrackingNtuple.cc

  constexpr unsigned int kPixelModuleId = 1;

  constexpr unsigned int max_blocks = 80;
  constexpr unsigned int max_connected_modules = 40;

  constexpr unsigned int n_max_pixel_segments_per_module = 500000;

  constexpr unsigned int n_max_pixel_md_per_modules = 2 * n_max_pixel_segments_per_module;

  constexpr unsigned int n_max_pixel_triplets = 5000;
  constexpr unsigned int n_max_pixel_quintuplets = 15000;

  constexpr unsigned int n_max_pixel_track_candidates = 300000;
  constexpr unsigned int n_max_nonpixel_track_candidates = 10000;

  constexpr unsigned int size_superbins = 45000;

  constexpr uint16_t kTCEmptyLowerModule = 0xFFFF;     // Sentinel for empty lowerModule index
  constexpr unsigned int kTCEmptyHitIdx = 0xFFFFFFFF;  // Sentinel for empty hit slots

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

  // Defining the constant host device variables right up here
  // Currently pixel tracks treated as LSs with 2 double layers (IT layers 1+2 and 3+4) and 4 hits. To be potentially handled better in the future.
  struct Params_Modules {
    using ArrayU16xMaxConnected = edm::StdArray<uint16_t, max_connected_modules>;
  };
  struct Params_pLS {
    static constexpr int kLayers = 2, kHits = 4;
    static constexpr int kEmbed = 6;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
    using ArrayFxEmbed = edm::StdArray<float, kEmbed>;
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
  struct Params_T4 {
    static constexpr int kLayers = 4, kHits = 8;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
  };
  struct Params_T5 {
    static constexpr int kLayers = 5, kHits = 10;
    static constexpr int kEmbed = 6;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
    using ArrayFxEmbed = edm::StdArray<float, kEmbed>;
  };
  struct Params_pT5 {
    static constexpr int kLayers = 7, kHits = 14;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUxHits = edm::StdArray<unsigned int, kHits>;
  };
  struct Params_TC {
    static constexpr int kLayers = 13;
    static constexpr int kHitsPerLayer = 2;
    // Number of layers resevered for pixel hits.
    static constexpr int kPixelLayerSlots = 2;
    static constexpr int kHits = kLayers * kHitsPerLayer;
    using ArrayU8xLayers = edm::StdArray<uint8_t, kLayers>;
    using ArrayU16xLayers = edm::StdArray<uint16_t, kLayers>;
    using ArrayUHitsPerLayer = edm::StdArray<unsigned int, kHitsPerLayer>;
    using ArrayUxHits = edm::StdArray<ArrayUHitsPerLayer, kLayers>;
  };

  using ArrayIx2 = edm::StdArray<int, 2>;
  using ArrayUx2 = edm::StdArray<unsigned int, 2>;

}  //namespace lst

#endif
