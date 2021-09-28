// -*- c++ -*-
#include <Eigen/Dense>

#include "RecoParticleFlow/PFClusterProducerCUDA/src/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
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
						 int* pfrechits_detId,
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
      pfrechits_x[i] = (float)0.5;
      pfrechits_y[i] = (float)0.6;
      pfrechits_z[i] = (float)0.7;

      


    }


  }// namespace pf_cuda_computation


} //  namespace hcal

namespace hcal {
  namespace reconstruction {
    
    void entryPoint_for_HBHEPFRecHit_Computation(hcal::reconstruction::OutputDataGPU const& HBHERecHits_asInput,
						 hcal::reconstruction::OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
						 cudaStream_t cudaStream) {
      //auto const recHits_en = HBHERecHits_asInput.recHits.energy.get();
      //size_t num_rechits = std::sizeof(recHits_en);
      //std::cout<<typeid(recHits_en).name()<<std::endl;
      hcal::pf_cuda_computation::convert_rechits_to_PFRechits<<<8, 4, 128, cudaStream>>>(
											 HBHERecHits_asInput.recHits.energy.get(),
											 HBHERecHits_asInput.recHits.chi2.get(),
											 HBHERecHits_asInput.recHits.energyM0.get(),
											 HBHERecHits_asInput.recHits.timeM0.get(),
											 HBHERecHits_asInput.recHits.did.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_depth.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_layer.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_caloId.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_detId.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_time.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_energy.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_pt2.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_x.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_y.get(),
											 HBHEPFRecHits_asOutput.PFRecHits.pfrh_z.get());

    }


  }// namespace reconstruction


} // namespace hcal
