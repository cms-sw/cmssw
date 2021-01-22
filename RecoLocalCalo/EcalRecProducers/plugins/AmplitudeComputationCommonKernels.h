#ifndef RecoLocalCalo_EcalRecProducers_plugins_AmplitudeComputationCommonKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_AmplitudeComputationCommonKernels_h

#include "Common.h"
#include "DeclsForKernels.h"
#include "EigenMatrixTypes_gpu.h"

class EcalPulseShape;
// this flag setting is applied to all of the cases
class EcalPulseCovariance;
class EcalUncalibratedRecHit;

namespace ecal {
  namespace multifit {

    ///
    /// assume kernel launch configuration is
    /// (MAXSAMPLES * nchannels, blocks)
    /// TODO: is there a point to split this kernel further to separate reductions
    ///
    __global__ void kernel_prep_1d_and_initialize(EcalPulseShape const* shapes_in,
                                                  uint16_t const* digis_in_eb,
                                                  uint32_t const* dids_eb,
                                                  uint16_t const* digis_in_ee,
                                                  uint32_t const* dids_ee,
                                                  SampleVector* amplitudes,
                                                  SampleVector* amplitudesForMinimizationEB,
                                                  SampleVector* amplitudesForMinimizationEE,
                                                  SampleGainVector* gainsNoise,
                                                  float const* mean_x1,
                                                  float const* mean_x12,
                                                  float const* rms_x12,
                                                  float const* mean_x6,
                                                  float const* gain6Over1,
                                                  float const* gain12Over6,
                                                  bool* hasSwitchToGain6,
                                                  bool* hasSwitchToGain1,
                                                  bool* isSaturated,
                                                  ::ecal::reco::StorageScalarType* energiesEB,
                                                  ::ecal::reco::StorageScalarType* energiesEE,
                                                  ::ecal::reco::StorageScalarType* chi2EB,
                                                  ::ecal::reco::StorageScalarType* chi2EE,
                                                  ::ecal::reco::StorageScalarType* pedestalEB,
                                                  ::ecal::reco::StorageScalarType* pedestalEE,
                                                  uint32_t* dids_outEB,
                                                  uint32_t* dids_outEE,
                                                  uint32_t* flagsEB,
                                                  uint32_t* flagsEE,
                                                  char* acState,
                                                  BXVectorType* bxs,
                                                  uint32_t const offsetForHashes,
                                                  uint32_t const offsetForInputs,
                                                  bool const gainSwitchUseMaxSampleEB,
                                                  bool const gainSwitchUseMaxSampleEE,
                                                  int const nchannels);

    ///
    /// assume kernel launch configuration is
    /// ([MAXSAMPLES, MAXSAMPLES], nchannels)
    ///
    __global__ void kernel_prep_2d(SampleGainVector const* gainNoise,
                                   uint32_t const* dids_eb,
                                   uint32_t const* dids_ee,
                                   float const* rms_x12,
                                   float const* rms_x6,
                                   float const* rms_x1,
                                   float const* gain12Over6,
                                   float const* gain6Over1,
                                   double const* G12SamplesCorrelationEB,
                                   double const* G6SamplesCorrelationEB,
                                   double const* G1SamplesCorrelationEB,
                                   double const* G12SamplesCorrelationEE,
                                   double const* G6SamplesCorrelationEE,
                                   double const* G1SamplesCorrelationEE,
                                   SampleMatrix* noisecov,
                                   PulseMatrixType* pulse_matrix,
                                   EcalPulseShape const* pulse_shape,
                                   bool const* hasSwitchToGain6,
                                   bool const* hasSwitchToGain1,
                                   bool const* isSaturated,
                                   uint32_t const offsetForHashes,
                                   uint32_t const offsetForInputs);

    __global__ void kernel_permute_results(SampleVector* amplitudes,
                                           BXVectorType const* activeBXs,
                                           ::ecal::reco::StorageScalarType* energies,
                                           char const* acState,
                                           int const nchannels);

///
/// Build an Ecal RecHit.
/// TODO: Use SoA data structures on the host directly
/// the reason for removing this from minimize kernel is to isolate the minimize +
/// again, building an aos rec hit involves strides... -> bad memory access pattern
///
#ifdef RUN_BUILD_AOS_RECHIT
    __global__ void kernel_build_rechit(
        float const* energies, float const* chi2s, uint32_t* dids, EcalUncalibratedRecHit* rechits, int nchannels);
#endif  // RUN_BUILD_AOS_RECHIT

  }  // namespace multifit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_AmplitudeComputationCommonKernels_h
