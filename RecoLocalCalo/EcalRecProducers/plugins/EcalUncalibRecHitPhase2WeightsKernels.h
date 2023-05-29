#ifndef RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_EcalUncalibRecHitPhase2WeightsKernels_h

#include "DeclsForKernelsPhase2.h"

namespace ecal {
  namespace weights {

    __global__ void Phase2WeightsKernel(uint16_t const* digis_in_eb,
                                        uint32_t const* dids_eb,
                                        ::ecal::reco::StorageScalarType* amplitudeEB,
                                        ::ecal::reco::StorageScalarType* amplitudeErrorEB,
                                        uint32_t* dids_outEB,
                                        int const nchannels,
                                        double const* weights_d,
                                        uint32_t* flagsEB);
  }  //namespace weights
}  //namespace ecal

#endif
