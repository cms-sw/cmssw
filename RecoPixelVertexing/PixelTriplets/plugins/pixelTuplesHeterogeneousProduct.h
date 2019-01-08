#ifndef RecoPixelVertexingPixelTripletsPixelTuplesHeterogeneousProduct_H
#define RecoPixelVertexingPixelTripletsPixelTuplesHeterogeneousProduct_H


#include <cstdint>
#include <vector>

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"

#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/CAConstants.h"

namespace siPixelRecHitsHeterogeneousProduct {
  struct HitsOnGPU;
}

namespace pixelTuplesHeterogeneousProduct {

  enum Quality: uint8_t { bad, dup, loose, strict, tight, highPurity };

  using CPUProduct = int; // dummy

  struct TuplesOnGPU {
    using Container = CAConstants::TuplesContainer;

    Container * tuples_d;
    AtomicPairCounter * apc_d;

    Rfit::helix_fit * helix_fit_results_d = nullptr;
    Quality * quality_d =  nullptr;

    TuplesOnGPU const * me_d = nullptr;

  };

  struct TuplesOnCPU {

    std::vector<uint32_t> indToEdm; // index of    tuple in reco tracks....


    using Container = TuplesOnGPU::Container;

    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hitsOnGPU_d = nullptr; // forwarding

    Container const * tuples = nullptr;

    Rfit::helix_fit const * helix_fit_results = nullptr;
    Quality * quality =  nullptr;

    TuplesOnGPU const * gpu_d = nullptr;
    uint32_t nTuples;
  };
 
  using GPUProduct = TuplesOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelTuples = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}



#endif

