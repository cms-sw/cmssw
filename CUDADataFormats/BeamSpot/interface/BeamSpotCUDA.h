#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

class BeamSpotCUDA {
public:
  // align to the CUDA L1 cache line size
  struct alignas(128) Data {
    float x, y, z;  // position
    // TODO: add covariance matrix

    float sigmaZ;
    float beamWidthX, beamWidthY;
    float dxdz, dydz;
    float emittanceX, emittanceY;
    float betaStar;
  };

  // default constructor, required by cms::cuda::Product<BeamSpotCUDA>
  BeamSpotCUDA() = default;

  // constructor that allocates cached device memory on the given CUDA stream
  BeamSpotCUDA(cudaStream_t stream) { data_d_ = cms::cuda::make_device_unique<Data>(stream); }

  // movable, non-copiable
  BeamSpotCUDA(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA(BeamSpotCUDA&&) = default;
  BeamSpotCUDA& operator=(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA& operator=(BeamSpotCUDA&&) = default;

  Data* data() { return data_d_.get(); }
  Data const* data() const { return data_d_.get(); }

  cms::cuda::device::unique_ptr<Data>& ptr() { return data_d_; }
  cms::cuda::device::unique_ptr<Data> const& ptr() const { return data_d_; }

private:
  cms::cuda::device::unique_ptr<Data> data_d_;
};

#endif  // CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
