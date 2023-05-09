#ifndef CUDADataFormats_Vertex_ZVertexHeterogeneousHost_H
#define CUDADataFormats_Vertex_ZVertexHeterogeneousHost_H

#include <cstdint>

#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"

// TODO: The class is created via inheritance of the PortableHostCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <int32_t S>
class ZVertexSoAHeterogeneousHost : public cms::cuda::PortableHostCollection<ZVertexSoAHeterogeneousLayout<>> {
public:
  explicit ZVertexSoAHeterogeneousHost() : cms::cuda::PortableHostCollection<ZVertexSoAHeterogeneousLayout<>>(S) {}

  // Constructor which specifies the SoA size and CUDA stream
  explicit ZVertexSoAHeterogeneousHost(cudaStream_t stream)
      : PortableHostCollection<ZVertexSoAHeterogeneousLayout<>>(S, stream) {}
};

using ZVertexSoAHost = ZVertexSoAHeterogeneousHost<zVertex::utilities::MAXTRACKS>;

#endif  // CUDADataFormats_Vertex_ZVertexHeterogeneousHost_H
