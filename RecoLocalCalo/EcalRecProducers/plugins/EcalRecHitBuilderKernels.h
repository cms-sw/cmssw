#ifndef RecoLocalCalo_EcalRecProducers_plugins_EcalRecHitBuilderKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_EcalRecHitBuilderKernels_h

//
// Builder of ECAL RecHits on GPU
//

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "Common.h"
#include "DeclsForKernels.h"

namespace ecal {
  namespace rechit {

    __global__ void kernel_create_ecal_rehit(
        // configuration
        int const* ChannelStatusToBeExcluded,
        uint32_t ChannelStatusToBeExcludedSize,
        bool killDeadChannels,
        bool const recoverEBIsolatedChannels,
        bool const recoverEEIsolatedChannels,
        bool const recoverEBVFE,
        bool const recoverEEVFE,
        bool const recoverEBFE,
        bool const recoverEEFE,
        // for flags setting
        int const* expanded_v_DB_reco_flags,
        uint32_t const* expanded_Sizes_v_DB_reco_flags,
        uint32_t const* expanded_flagbit_v_DB_reco_flags,
        uint32_t expanded_v_DB_reco_flagsSize,
        uint32_t flagmask,
        // conditions
        float const* adc2gev,
        float const* intercalib,
        uint16_t const* status,
        float const* apdpnrefs,
        float const* alphas,
        // input for transparency corrections
        float const* p1,
        float const* p2,
        float const* p3,
        edm::TimeValue_t const* t1,
        edm::TimeValue_t const* t2,
        edm::TimeValue_t const* t3,
        // input for linear corrections
        float const* lp1,
        float const* lp2,
        float const* lp3,
        edm::TimeValue_t const* lt1,
        edm::TimeValue_t const* lt2,
        edm::TimeValue_t const* lt3,
        // time, used for time dependent corrections
        edm::TimeValue_t const event_time,
        // input
        uint32_t const* did_eb,
        uint32_t const* did_ee,
        ::ecal::reco::StorageScalarType const* amplitude_eb,  // in adc counts
        ::ecal::reco::StorageScalarType const* amplitude_ee,  // in adc counts
        ::ecal::reco::StorageScalarType const* time_eb,
        ::ecal::reco::StorageScalarType const* time_ee,
        ::ecal::reco::StorageScalarType const* chi2_eb,
        ::ecal::reco::StorageScalarType const* chi2_ee,
        uint32_t const* flags_eb,
        uint32_t const* flags_ee,
        // output
        uint32_t* did,
        ::ecal::reco::StorageScalarType* energy,  // in energy [GeV]
        ::ecal::reco::StorageScalarType* time,
        ::ecal::reco::StorageScalarType* chi2,
        uint32_t* flagBits,
        uint32_t* extra,
        int const nchannels,
        uint32_t const nChannelsBarrel,
        uint32_t const offsetForHashes);

    // host version, to be called by the plugin

    void create_ecal_rehit(EventInputDataGPU const& eventInputGPU,
                           EventOutputDataGPU& eventOutputGPU,
                           ConditionsProducts const& conditions,
                           ConfigurationParameters const& configParameters,
                           uint32_t const nChannelsBarrel,
                           edm::TimeValue_t const event_time,
                           cudaStream_t cudaStream);

  }  // namespace rechit

}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_EcalRecHitBuilderKernels_h
