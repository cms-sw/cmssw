#ifndef RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsKernels_h

#include "DeclsForKernelsPh2WeightsGPU.h"
#include "EigenMatrixTypes_gpu.h"

#include "DeclsForKernelsPh2WeightsGPU.h"

class EcalUncalibratedRecHit;

namespace ecal {
  namespace weights {

    __global__ void Phase2WeightsKernel(uint16_t const* digis_in_eb,
                                        uint32_t const* dids_eb,
                                        ::ecal::reco::StorageScalarType* amplitudeEB,
                                        uint32_t* dids_outEB,
                                        int const nchannels,
                                        double* weights_d,
                                        uint32_t* flagsEB
                                      //  ,uint16_t* Debug
                                        );
  } //namespace weights
} //namespace ecal

#endif