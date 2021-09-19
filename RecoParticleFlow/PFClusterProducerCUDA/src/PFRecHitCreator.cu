// -*- c++ -*-
#include <Eigen/Dense>

#include "RecoParticleFlow/PFClusterProducerCUDA/src/DeclsForKernels.h"

namespace hcal {
  namespace pf_cuda_computation {
    __global__ void convert_rechits_to_PFRechits(float const* recHits_energy,
						 float const* recHits_chi2,
						 float const* recHits_energyM0,
						 float const* recHits_timeM0,
						 uint32_t const* recHits_did,
						 int* pfrechits_depth,
						 int* pfrechits_layer,
						 int* pfrechits_caloId,
						 uint32_t* pfrechits_detId,
						 float* pfrechits_time,
						 float* pfrechits_energy,
						 float* pfrechits_pt2,
						 float* pfrechits_x,
						 float* pfrechits_y,
						 float* pfrechits_z) {

      
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      pfrechits_time[i] = recHits_timeM0[i];
      pfrechits_energy[i] = recHits_energyM0[i];
      pfrechits_detId[i] = recHits_did[i];
      


    }


  }// namespace pf_cuda_computation


} //  namespace hcal

namespace hcal {
  namespace reconstruction {
    
    void entryPoint_for_HBHEPFRecHit_Computation(OutputDataGPU const& HBHERecHits_asInput,
						 OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
						 cudaStream_t cudaStream) {
      size_t num_rechits = HBHERecHits_asInput.energy.size();
      


    }


  }// namespace reconstruction


} // namespace hcal
