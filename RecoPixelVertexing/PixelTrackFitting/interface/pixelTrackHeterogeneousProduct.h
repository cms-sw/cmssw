#ifndef RecoPixelVertexing_PixelTrackFitting_interface_pixelTrackHeterogeneousProduct_h
#define RecoPixelVertexing_PixelTrackFitting_interface_pixelTrackHeterogeneousProduct_h

#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"

namespace pixelTrackHeterogeneousProduct {

  static constexpr uint32_t maxTracks() { return 10000;}

  using CPUProduct = int; // dummy

  struct TracksOnGPU {

    Rfit::helix_fit * helix_fit_results_d;

    TracksOnGPU const * me_d = nullptr;

  };

  struct TracksOnCPU {

    Rfit::helix_fit * helix_fit_results;
    TracksOnGPU const * gpu_d = nullptr;
    uint32_t nTracks;
  };

  using GPUProduct = TracksOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelTuples = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}

}

#endif
