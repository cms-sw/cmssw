#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#include <cuda_runtime.h>

#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaMemoryPool.h"

class BeamSpotCUDA {
public:
 
  using buffer = memoryPool::buffer<BeamSpotPOD>;

  // default constructor, required by cms::cuda::Product<BeamSpotCUDA>
  BeamSpotCUDA() = default;

  // constructor that allocates cached device memory on the given CUDA stream
  BeamSpotCUDA(cudaStream_t stream) { data_d_ = memoryPool::cuda::make_buffer<BeamSpotPOD>(1,stream, memoryPool::onDevice); }

  // movable, non-copiable
  BeamSpotCUDA(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA(BeamSpotCUDA&&) = default;
  BeamSpotCUDA& operator=(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA& operator=(BeamSpotCUDA&&) = default;

  BeamSpotPOD* data() { return data_d_.get(); }
  BeamSpotPOD const* data() const { return data_d_.get(); }

  buffer & ptr() { return data_d_; }
  buffer const& ptr() const { return data_d_; }

private:
  buffer data_d_;
};

#endif  // CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
