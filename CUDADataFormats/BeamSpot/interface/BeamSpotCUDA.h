#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#include <cuda_runtime.h>

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

class BeamSpotCUDA {
public:
  // default constructor, required by cms::cuda::Product<BeamSpotCUDA>
  BeamSpotCUDA() = default;

  // constructor that allocates cached device memory on the given CUDA stream
  BeamSpotCUDA(cudaStream_t stream) { data_d_ = cms::cuda::make_device_unique<BeamSpotPOD>(stream); }

  // movable, non-copiable
  BeamSpotCUDA(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA(BeamSpotCUDA&&) = default;
  BeamSpotCUDA& operator=(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA& operator=(BeamSpotCUDA&&) = default;

  BeamSpotPOD* data() { return data_d_.get(); }
  BeamSpotPOD const* data() const { return data_d_.get(); }

  cms::cuda::device::unique_ptr<BeamSpotPOD>& ptr() { return data_d_; }
  cms::cuda::device::unique_ptr<BeamSpotPOD> const& ptr() const { return data_d_; }

private:
  cms::cuda::device::unique_ptr<BeamSpotPOD> data_d_;
};

#endif  // CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
