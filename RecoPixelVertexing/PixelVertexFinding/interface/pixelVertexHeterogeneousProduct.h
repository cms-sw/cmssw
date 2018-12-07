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
    uint16_t * sortInd;
    int32_t * ivtx_d; // this should be indexed with the original tracks, not the reduced set (oops)
  };


  struct VerticesOnCPU {
    VerticesOnCPU() = default;

    explicit VerticesOnCPU(uint32_t nvtx, uint32_t ntrks) :
      z(nvtx),
      zerr(nvtx),
      ivtx(ntrks),
      nVertices(nvtx)
    { }

    std::vector<float,    CUDAHostAllocator<float>> z,zerr, chi2;
    std::vector<int16_t, CUDAHostAllocator<uint16_t>> sortInd;
    std::vector<int32_t, CUDAHostAllocator<int32_t>> ivtx;

    uint32_t nVertices=0;
    VerticesOnGPU const * gpu_d = nullptr;
  };


  using GPUProduct = VerticesOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelVertices = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                              heterogeneous::GPUCudaProduct<GPUProduct> >;
}
#endif
