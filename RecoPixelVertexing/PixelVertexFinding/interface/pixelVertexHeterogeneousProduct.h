#ifndef RecoPixelVertexing_PixelVertexFinding_interface_pixelVertexHeterogeneousProduct_h
#define RecoPixelVertexing_PixelVertexFinding_interface_pixelVertexHeterogeneousProduct_h

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

namespace pixelVertexHeterogeneousProduct {

  struct CPUProduct {
    reco::VertexCollection collection;
  };

  struct VerticesOnCPU {
    VerticesOnCPU() = default;

    float const *z, *zerr, *chi2;
    int16_t const *sortInd;
    int32_t const *ivtx;
    uint16_t const *itrk;

    uint32_t nVertices = 0;
    uint32_t nTracks = 0;
  };

  using GPUProduct = VerticesOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelVertices =
      HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>, heterogeneous::GPUCudaProduct<GPUProduct> >;

}  // namespace pixelVertexHeterogeneousProduct

#endif  // RecoPixelVertexing_PixelVertexFinding_interface_pixelVertexHeterogeneousProduct_h
