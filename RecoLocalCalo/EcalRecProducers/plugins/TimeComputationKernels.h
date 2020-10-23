#ifndef RecoLocalCalo_EcalRecProducers_plugins_TimeComputationKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_TimeComputationKernels_h

#include <iostream>
#include <limits>

#include <cuda.h>

#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"

#include "Common.h"
#include "DeclsForKernels.h"
#include "EigenMatrixTypes_gpu.h"

//#define DEBUG

//#define ECAL_RECO_CUDA_DEBUG

namespace ecal {
  namespace multifit {

    __global__ void kernel_time_compute_nullhypot(SampleVector::Scalar const* sample_values,
                                                  SampleVector::Scalar const* sample_value_errors,
                                                  bool const* useless_sample_values,
                                                  SampleVector::Scalar* chi2s,
                                                  SampleVector::Scalar* sum0s,
                                                  SampleVector::Scalar* sumAAs,
                                                  int const nchannels);
    //
    // launch ctx parameters are
    // 45 threads per channel, X channels per block, Y blocks
    // 45 comes from: 10 samples for i <- 0 to 9 and for j <- i+1 to 9
    // TODO: it might be much beter to use 32 threads per channel instead of 45
    // to simplify the synchronization
    //
    __global__ void kernel_time_compute_makeratio(SampleVector::Scalar const* sample_values,
                                                  SampleVector::Scalar const* sample_value_errors,
                                                  uint32_t const* dids_eb,
                                                  uint32_t const* dids_ee,
                                                  bool const* useless_sample_values,
                                                  char const* pedestal_nums,
                                                  ConfigurationParameters::type const* amplitudeFitParametersEB,
                                                  ConfigurationParameters::type const* amplitudeFitParametersEE,
                                                  ConfigurationParameters::type const* timeFitParametersEB,
                                                  ConfigurationParameters::type const* timeFitParametersEE,
                                                  SampleVector::Scalar const* sumAAsNullHypot,
                                                  SampleVector::Scalar const* sum0sNullHypot,
                                                  SampleVector::Scalar* tMaxAlphaBetas,
                                                  SampleVector::Scalar* tMaxErrorAlphaBetas,
                                                  SampleVector::Scalar* g_accTimeMax,
                                                  SampleVector::Scalar* g_accTimeWgt,
                                                  TimeComputationState* g_state,
                                                  unsigned int const timeFitParameters_sizeEB,
                                                  unsigned int const timeFitParameters_sizeEE,
                                                  ConfigurationParameters::type const timeFitLimits_firstEB,
                                                  ConfigurationParameters::type const timeFitLimits_firstEE,
                                                  ConfigurationParameters::type const timeFitLimits_secondEB,
                                                  ConfigurationParameters::type const timeFitLimits_secondEE,
                                                  int const nchannels,
                                                  uint32_t const offsetForInputs);

    /// launch ctx parameters are
    /// 10 threads per channel, N channels per block, Y blocks
    /// TODO: do we need to keep the state around or can be removed?!
    //#define DEBUG_FINDAMPLCHI2_AND_FINISH
    __global__ void kernel_time_compute_findamplchi2_and_finish(
        SampleVector::Scalar const* sample_values,
        SampleVector::Scalar const* sample_value_errors,
        uint32_t const* dids_eb,
        uint32_t const* dids_ee,
        bool const* useless_samples,
        SampleVector::Scalar const* g_tMaxAlphaBeta,
        SampleVector::Scalar const* g_tMaxErrorAlphaBeta,
        SampleVector::Scalar const* g_accTimeMax,
        SampleVector::Scalar const* g_accTimeWgt,
        ConfigurationParameters::type const* amplitudeFitParametersEB,
        ConfigurationParameters::type const* amplitudeFitParametersEE,
        SampleVector::Scalar const* sumAAsNullHypot,
        SampleVector::Scalar const* sum0sNullHypot,
        SampleVector::Scalar const* chi2sNullHypot,
        TimeComputationState* g_state,
        SampleVector::Scalar* g_ampMaxAlphaBeta,
        SampleVector::Scalar* g_ampMaxError,
        SampleVector::Scalar* g_timeMax,
        SampleVector::Scalar* g_timeError,
        int const nchannels,
        uint32_t const offsetForInputs);

    __global__ void kernel_time_compute_fixMGPAslew(uint16_t const* digis_eb,
                                                    uint16_t const* digis_ee,
                                                    SampleVector::Scalar* sample_values,
                                                    SampleVector::Scalar* sample_value_errors,
                                                    bool* useless_sample_values,
                                                    unsigned int const sample_mask,
                                                    int const nchannels,
                                                    uint32_t const offsetForInputs);

    __global__ void kernel_time_compute_ampl(SampleVector::Scalar const* sample_values,
                                             SampleVector::Scalar const* sample_value_errors,
                                             uint32_t const* dids_eb,
                                             uint32_t const* dids_ed,
                                             bool const* useless_samples,
                                             SampleVector::Scalar const* g_timeMax,
                                             SampleVector::Scalar const* amplitudeFitParametersEB,
                                             SampleVector::Scalar const* amplitudeFitParametersEE,
                                             SampleVector::Scalar* g_amplitudeMax,
                                             int const nchannels,
                                             uint32_t const offsetForInputs);

    //#define ECAL_RECO_CUDA_TC_INIT_DEBUG
    __global__ void kernel_time_computation_init(uint16_t const* digis_eb,
                                                 uint32_t const* dids_eb,
                                                 uint16_t const* digis_ee,
                                                 uint32_t const* dids_ee,
                                                 float const* rms_x12,
                                                 float const* rms_x6,
                                                 float const* rms_x1,
                                                 float const* mean_x12,
                                                 float const* mean_x6,
                                                 float const* mean_x1,
                                                 float const* gain12Over6,
                                                 float const* gain6Over1,
                                                 SampleVector::Scalar* sample_values,
                                                 SampleVector::Scalar* sample_value_errors,
                                                 SampleVector::Scalar* ampMaxError,
                                                 bool* useless_sample_values,
                                                 char* pedestal_nums,
                                                 uint32_t const offsetForHashes,
                                                 uint32_t const offsetForInputs,
                                                 unsigned int const sample_maskEB,
                                                 unsigned int const sample_maskEE,
                                                 int nchannels);

    ///
    /// launch context parameters: 1 thread per channel
    ///
    //#define DEBUG_TIME_CORRECTION
    __global__ void kernel_time_correction_and_finalize(
        //        SampleVector::Scalar const* g_amplitude,
        ::ecal::reco::StorageScalarType const* g_amplitudeEB,
        ::ecal::reco::StorageScalarType const* g_amplitudeEE,
        uint16_t const* digis_eb,
        uint32_t const* dids_eb,
        uint16_t const* digis_ee,
        uint32_t const* dids_ee,
        float const* amplitudeBinsEB,
        float const* amplitudeBinsEE,
        float const* shiftBinsEB,
        float const* shiftBinsEE,
        SampleVector::Scalar const* g_timeMax,
        SampleVector::Scalar const* g_timeError,
        float const* g_rms_x12,
        float const* timeCalibConstant,
        ::ecal::reco::StorageScalarType* g_jitterEB,
        ::ecal::reco::StorageScalarType* g_jitterEE,
        ::ecal::reco::StorageScalarType* g_jitterErrorEB,
        ::ecal::reco::StorageScalarType* g_jitterErrorEE,
        uint32_t* flagsEB,
        uint32_t* flagsEE,
        int const amplitudeBinsSizeEB,
        int const amplitudeBinsSizeEE,
        ConfigurationParameters::type const timeConstantTermEB,
        ConfigurationParameters::type const timeConstantTermEE,
        float const offsetTimeValueEB,
        float const offsetTimeValueEE,
        ConfigurationParameters::type const timeNconstEB,
        ConfigurationParameters::type const timeNconstEE,
        ConfigurationParameters::type const amplitudeThresholdEB,
        ConfigurationParameters::type const amplitudeThresholdEE,
        ConfigurationParameters::type const outOfTimeThreshG12pEB,
        ConfigurationParameters::type const outOfTimeThreshG12pEE,
        ConfigurationParameters::type const outOfTimeThreshG12mEB,
        ConfigurationParameters::type const outOfTimeThreshG12mEE,
        ConfigurationParameters::type const outOfTimeThreshG61pEB,
        ConfigurationParameters::type const outOfTimeThreshG61pEE,
        ConfigurationParameters::type const outOfTimeThreshG61mEB,
        ConfigurationParameters::type const outOfTimeThreshG61mEE,
        uint32_t const offsetForHashes,
        uint32_t const offsetForInputs,
        int const nchannels);

  }  // namespace multifit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_TimeComputationKernels_h
