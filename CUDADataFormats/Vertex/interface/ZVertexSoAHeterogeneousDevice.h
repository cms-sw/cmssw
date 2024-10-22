#ifndef CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H
#define CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H

#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
template <int32_t S>
class ZVertexSoAHeterogeneousDevice : public cms::cuda::PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>> {
public:
  ZVertexSoAHeterogeneousDevice() = default;  // cms::cuda::Product needs this

  // Constructor which specifies the SoA size
  explicit ZVertexSoAHeterogeneousDevice(cudaStream_t stream)
      : PortableDeviceCollection<ZVertexSoAHeterogeneousLayout<>>(S, stream) {}
};

using ZVertexSoADevice = ZVertexSoAHeterogeneousDevice<zVertex::utilities::MAXTRACKS>;

#endif  // CUDADataFormats_Vertex_ZVertexHeterogeneousDevice_H
