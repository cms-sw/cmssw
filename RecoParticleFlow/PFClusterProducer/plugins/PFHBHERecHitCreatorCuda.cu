// -*- c++ -*-
#include <Eigen/Dense>

#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
namespace hcal {
  namespace pf_cuda_computation {
    __global__ void convert_rechits_to_PFRechits(int size,
                         float const* recHits_energy,
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
      for (int j = i; j < size; j += blockDim.x*gridDim.x) {

          pfrechits_time[j] = recHits_timeM0[j];
          pfrechits_energy[j] = recHits_energyM0[j];
          pfrechits_detId[j] = recHits_did[j];
          pfrechits_x[j] = (float)0.5;
          pfrechits_y[j] = (float)0.6;
          pfrechits_z[j] = (float)0.7;

      

      }
    }


  }// namespace pf_cuda_computation


} //  namespace hcal

namespace hcal {
  namespace reconstruction {
    //void entryPoint_for_HBHEPFRecHit_Computation(hcal::reconstruction::OutputDataGPU const& HBHERecHits_asInput,
    void entryPoint_for_PFComputation(
        ::hcal::RecHitCollection<calo::common::DevStoragePolicy> const& HBHERecHits_asInput,
        ::hcal::reconstruction::OutputPFRecHitDataGPU& HBHEPFRecHits_asOutput,
		cudaStream_t cudaStream) {
      
      int nRH = HBHERecHits_asInput.size;   // Number of input rechits
      printf("Now in entrypoint_for_HBHEPFRecHit_Computation\n");
      //auto const recHits_en = HBHERecHits_asInput.recHits.energy.get();
      //size_t num_rechits = std::sizeof(recHits_en);
      //std::cout<<typeid(recHits_en).name()<<std::endl;
      ::hcal::pf_cuda_computation::convert_rechits_to_PFRechits<<<(nRH+31)/32, 128, 0, cudaStream>>>(
											 nRH,
                                             HBHERecHits_asInput.energy.get(),
											 HBHERecHits_asInput.chi2.get(),
											 HBHERecHits_asInput.energyM0.get(),
											 HBHERecHits_asInput.timeM0.get(),
											 HBHERecHits_asInput.did.get(),
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
