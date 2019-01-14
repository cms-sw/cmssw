#ifndef RecoPixelVertexing_PixelVertexFinding_pixelVertexHeterogeneousProduct_H
#define RecoPixelVertexing_PixelVertexFinding_pixelVertexHeterogeneousProduct_H

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"

namespace pixelVertexHeterogeneousProduct {

  struct CPUProduct {
    reco::VertexCollection collection;
  };

  struct VerticesOnGPU{
    float * z_d;
    float * zerr_d;
    float * chi2_d;
    uint16_t * sortInd_d;
    int32_t * ivtx_d; // this should be indexed with the original tracks, not the reduced set (oops)
  };


  struct VerticesOnCPU {
    VerticesOnCPU() = default;

    float const *z, *zerr, *chi2;
    int16_t const * sortInd;
    int32_t const * ivtx;
    uint16_t const * itrk;

    uint32_t nVertices=0;
    uint32_t nTracks=0;
    VerticesOnGPU const * gpu_d = nullptr;
  };


  using GPUProduct = VerticesOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelVertices = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                              heterogeneous::GPUCudaProduct<GPUProduct> >;
}
#endif
