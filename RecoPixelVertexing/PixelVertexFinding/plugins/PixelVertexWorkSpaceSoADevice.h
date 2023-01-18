#ifndef RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoADevice_h
#define RecoPixelVertexing_PixelVertexFinding_PixelVertexWorkSpaceSoADevice_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "RecoPixelVertexing/PixelVertexFinding/plugins/PixelVertexWorkSpaceUtilities.h"

template <int32_t S>
class PixelVertexWorkSpaceSoADevice : public cms::cuda::PortableDeviceCollection<PixelVertexWSSoALayout<>> {
public:
  PixelVertexWorkSpaceSoADevice() = default;

  // Constructor which specifies the SoA size and CUDA stream
  explicit PixelVertexWorkSpaceSoADevice(cudaStream_t stream)
      : PortableDeviceCollection<PixelVertexWSSoALayout<>>(S, stream) {}
};

namespace gpuVertexFinder {
  namespace workSpace {
    using PixelVertexWorkSpaceSoADevice = PixelVertexWorkSpaceSoADevice<zVertex::utilities::MAXTRACKS>;
  }
}  // namespace gpuVertexFinder
#endif
