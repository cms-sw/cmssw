#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include <cuda_runtime.h>

class BeamSpotCUDA {
public:
  // alignas(128) doesn't really make sense as there is only one
  // beamspot per event?
  struct Data {
    float x, y, z;  // position
    // TODO: add covariance matrix

    float sigmaZ;
    float beamWidthX, beamWidthY;
    float dxdz, dydz;
    float emittanceX, emittanceY;
    float betaStar;
  };

  BeamSpotCUDA() = default;
  BeamSpotCUDA(Data const* data_h, cudaStream_t stream);

  Data const* data() const { return data_d_.get(); }

private:
  cms::cuda::device::unique_ptr<Data> data_d_;
};

#endif
