#include <iostream>
#include <limits>

#include <cuda.h>

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "AmplitudeComputationCommonKernels.h"
#include "AmplitudeComputationKernels.h"
#include "Common.h"
#include "EcalUncalibRecHitMultiFitAlgoGPU.h"
#include "TimeComputationKernels.h"

//#define DEBUG

//#define ECAL_RECO_CUDA_DEBUG

namespace ecal {
  namespace multifit {

    void entryPoint(EventInputDataGPU const& eventInputGPU,
                    EventOutputDataGPU& eventOutputGPU,
                    EventDataForScratchGPU& scratch,
                    ConditionsProducts const& conditions,
                    ConfigurationParameters const& configParameters,
                    cudaStream_t cudaStream) {
      using digis_type = std::vector<uint16_t>;
      using dids_type = std::vector<uint32_t>;
      // accodring to the cpu setup  //----> hardcoded
      bool const gainSwitchUseMaxSampleEB = true;
      // accodring to the cpu setup  //----> hardcoded
      bool const gainSwitchUseMaxSampleEE = false;

      uint32_t const offsetForHashes = conditions.offsetForHashes;
      uint32_t const offsetForInputs = eventInputGPU.ebDigis.size;
      unsigned int totalChannels = eventInputGPU.ebDigis.size + eventInputGPU.eeDigis.size;

      //
      // 1d preparation kernel
      //
      unsigned int nchannels_per_block = 32;
      unsigned int threads_1d = 10 * nchannels_per_block;
      unsigned int blocks_1d = threads_1d > 10 * totalChannels ? 1 : (totalChannels * 10 + threads_1d - 1) / threads_1d;
      int shared_bytes = nchannels_per_block * EcalDataFrame::MAXSAMPLES *
                         (sizeof(bool) + sizeof(bool) + sizeof(bool) + sizeof(bool) + sizeof(char) + sizeof(bool));
      kernel_prep_1d_and_initialize<<<blocks_1d, threads_1d, shared_bytes, cudaStream>>>(
          conditions.pulseShapes.values,
          eventInputGPU.ebDigis.data.get(),
          eventInputGPU.ebDigis.ids.get(),
          eventInputGPU.eeDigis.data.get(),
          eventInputGPU.eeDigis.ids.get(),
          (SampleVector*)scratch.samples.get(),
          (SampleVector*)eventOutputGPU.recHitsEB.amplitudesAll.get(),
          (SampleVector*)eventOutputGPU.recHitsEE.amplitudesAll.get(),
          (SampleGainVector*)scratch.gainsNoise.get(),
          conditions.pedestals.mean_x1,
          conditions.pedestals.mean_x12,
          conditions.pedestals.rms_x12,
          conditions.pedestals.mean_x6,
          conditions.gainRatios.gain6Over1,
          conditions.gainRatios.gain12Over6,
          scratch.hasSwitchToGain6.get(),
          scratch.hasSwitchToGain1.get(),
          scratch.isSaturated.get(),
          eventOutputGPU.recHitsEB.amplitude.get(),
          eventOutputGPU.recHitsEE.amplitude.get(),
          eventOutputGPU.recHitsEB.chi2.get(),
          eventOutputGPU.recHitsEE.chi2.get(),
          eventOutputGPU.recHitsEB.pedestal.get(),
          eventOutputGPU.recHitsEE.pedestal.get(),
          eventOutputGPU.recHitsEB.did.get(),
          eventOutputGPU.recHitsEE.did.get(),
          eventOutputGPU.recHitsEB.flags.get(),
          eventOutputGPU.recHitsEE.flags.get(),
          scratch.acState.get(),
          (BXVectorType*)scratch.activeBXs.get(),
          offsetForHashes,
          offsetForInputs,
          gainSwitchUseMaxSampleEB,
          gainSwitchUseMaxSampleEE,
          totalChannels);
      cudaCheck(cudaGetLastError());

      //
      // 2d preparation kernel
      //
      int blocks_2d = totalChannels;
      dim3 threads_2d{10, 10};
      kernel_prep_2d<<<blocks_2d, threads_2d, 0, cudaStream>>>((SampleGainVector*)scratch.gainsNoise.get(),
                                                               eventInputGPU.ebDigis.ids.get(),
                                                               eventInputGPU.eeDigis.ids.get(),
                                                               conditions.pedestals.rms_x12,
                                                               conditions.pedestals.rms_x6,
                                                               conditions.pedestals.rms_x1,
                                                               conditions.gainRatios.gain12Over6,
                                                               conditions.gainRatios.gain6Over1,
                                                               conditions.samplesCorrelation.EBG12SamplesCorrelation,
                                                               conditions.samplesCorrelation.EBG6SamplesCorrelation,
                                                               conditions.samplesCorrelation.EBG1SamplesCorrelation,
                                                               conditions.samplesCorrelation.EEG12SamplesCorrelation,
                                                               conditions.samplesCorrelation.EEG6SamplesCorrelation,
                                                               conditions.samplesCorrelation.EEG1SamplesCorrelation,
                                                               (SampleMatrix*)scratch.noisecov.get(),
                                                               (PulseMatrixType*)scratch.pulse_matrix.get(),
                                                               conditions.pulseShapes.values,
                                                               scratch.hasSwitchToGain6.get(),
                                                               scratch.hasSwitchToGain1.get(),
                                                               scratch.isSaturated.get(),
                                                               offsetForHashes,
                                                               offsetForInputs);
      cudaCheck(cudaGetLastError());

      // run minimization kernels
      v1::minimization_procedure(eventInputGPU, eventOutputGPU, scratch, conditions, configParameters, cudaStream);

      if (configParameters.shouldRunTimingComputation) {
        //
        // TODO: this guy can run concurrently with other kernels,
        // there is no dependence on the order of execution
        //
        unsigned int threads_time_init = threads_1d;
        unsigned int blocks_time_init = blocks_1d;
        int sharedBytesInit = 2 * threads_time_init * sizeof(SampleVector::Scalar);
        kernel_time_computation_init<<<blocks_time_init, threads_time_init, sharedBytesInit, cudaStream>>>(
            eventInputGPU.ebDigis.data.get(),
            eventInputGPU.ebDigis.ids.get(),
            eventInputGPU.eeDigis.data.get(),
            eventInputGPU.eeDigis.ids.get(),
            conditions.pedestals.rms_x12,
            conditions.pedestals.rms_x6,
            conditions.pedestals.rms_x1,
            conditions.pedestals.mean_x12,
            conditions.pedestals.mean_x6,
            conditions.pedestals.mean_x1,
            conditions.gainRatios.gain12Over6,
            conditions.gainRatios.gain6Over1,
            scratch.sample_values.get(),
            scratch.sample_value_errors.get(),
            scratch.ampMaxError.get(),
            scratch.useless_sample_values.get(),
            scratch.pedestal_nums.get(),
            offsetForHashes,
            offsetForInputs,
            conditions.sampleMask.getEcalSampleMaskRecordEB(),
            conditions.sampleMask.getEcalSampleMaskRecordEE(),
            totalChannels);
        cudaCheck(cudaGetLastError());

        //
        // TODO: small kernel only for EB. It needs to be checked if
        /// fusing such small kernels is beneficial in here
        //
        // we are running only over EB digis
        // therefore we need to create threads/blocks only for that
        unsigned int const threadsFixMGPA = threads_1d;
        unsigned int const blocksFixMGPA =
            threadsFixMGPA > 10 * eventInputGPU.ebDigis.size
                ? 1
                : (10 * eventInputGPU.ebDigis.size + threadsFixMGPA - 1) / threadsFixMGPA;
        kernel_time_compute_fixMGPAslew<<<blocksFixMGPA, threadsFixMGPA, 0, cudaStream>>>(
            eventInputGPU.ebDigis.data.get(),
            eventInputGPU.eeDigis.data.get(),
            scratch.sample_values.get(),
            scratch.sample_value_errors.get(),
            scratch.useless_sample_values.get(),
            conditions.sampleMask.getEcalSampleMaskRecordEB(),
            totalChannels,
            offsetForInputs);
        cudaCheck(cudaGetLastError());

        int sharedBytes = EcalDataFrame::MAXSAMPLES * nchannels_per_block * 4 * sizeof(SampleVector::Scalar);
        auto const threads_nullhypot = threads_1d;
        auto const blocks_nullhypot = blocks_1d;
        kernel_time_compute_nullhypot<<<blocks_nullhypot, threads_nullhypot, sharedBytes, cudaStream>>>(
            scratch.sample_values.get(),
            scratch.sample_value_errors.get(),
            scratch.useless_sample_values.get(),
            scratch.chi2sNullHypot.get(),
            scratch.sum0sNullHypot.get(),
            scratch.sumAAsNullHypot.get(),
            totalChannels);
        cudaCheck(cudaGetLastError());

        unsigned int nchannels_per_block_makeratio = 10;
        unsigned int threads_makeratio = 45 * nchannels_per_block_makeratio;
        unsigned int blocks_makeratio = threads_makeratio > 45 * totalChannels
                                            ? 1
                                            : (totalChannels * 45 + threads_makeratio - 1) / threads_makeratio;
        int sharedBytesMakeRatio = 5 * threads_makeratio * sizeof(SampleVector::Scalar);
        kernel_time_compute_makeratio<<<blocks_makeratio, threads_makeratio, sharedBytesMakeRatio, cudaStream>>>(
            scratch.sample_values.get(),
            scratch.sample_value_errors.get(),
            eventInputGPU.ebDigis.ids.get(),
            eventInputGPU.eeDigis.ids.get(),
            scratch.useless_sample_values.get(),
            scratch.pedestal_nums.get(),
            configParameters.amplitudeFitParametersEB,
            configParameters.amplitudeFitParametersEE,
            configParameters.timeFitParametersEB,
            configParameters.timeFitParametersEE,
            scratch.sumAAsNullHypot.get(),
            scratch.sum0sNullHypot.get(),
            scratch.tMaxAlphaBetas.get(),
            scratch.tMaxErrorAlphaBetas.get(),
            scratch.accTimeMax.get(),
            scratch.accTimeWgt.get(),
            scratch.tcState.get(),
            configParameters.timeFitParametersSizeEB,
            configParameters.timeFitParametersSizeEE,
            configParameters.timeFitLimitsFirstEB,
            configParameters.timeFitLimitsFirstEE,
            configParameters.timeFitLimitsSecondEB,
            configParameters.timeFitLimitsSecondEE,
            totalChannels,
            offsetForInputs);
        cudaCheck(cudaGetLastError());

        auto const threads_findamplchi2 = threads_1d;
        auto const blocks_findamplchi2 = blocks_1d;
        int const sharedBytesFindAmplChi2 = 2 * threads_findamplchi2 * sizeof(SampleVector::Scalar);
        kernel_time_compute_findamplchi2_and_finish<<<blocks_findamplchi2,
                                                      threads_findamplchi2,
                                                      sharedBytesFindAmplChi2,
                                                      cudaStream>>>(scratch.sample_values.get(),
                                                                    scratch.sample_value_errors.get(),
                                                                    eventInputGPU.ebDigis.ids.get(),
                                                                    eventInputGPU.eeDigis.ids.get(),
                                                                    scratch.useless_sample_values.get(),
                                                                    scratch.tMaxAlphaBetas.get(),
                                                                    scratch.tMaxErrorAlphaBetas.get(),
                                                                    scratch.accTimeMax.get(),
                                                                    scratch.accTimeWgt.get(),
                                                                    configParameters.amplitudeFitParametersEB,
                                                                    configParameters.amplitudeFitParametersEE,
                                                                    scratch.sumAAsNullHypot.get(),
                                                                    scratch.sum0sNullHypot.get(),
                                                                    scratch.chi2sNullHypot.get(),
                                                                    scratch.tcState.get(),
                                                                    scratch.ampMaxAlphaBeta.get(),
                                                                    scratch.ampMaxError.get(),
                                                                    scratch.timeMax.get(),
                                                                    scratch.timeError.get(),
                                                                    totalChannels,
                                                                    offsetForInputs);
        cudaCheck(cudaGetLastError());

        auto const threads_timecorr = 32;
        auto const blocks_timecorr =
            threads_timecorr > totalChannels ? 1 : (totalChannels + threads_timecorr - 1) / threads_timecorr;
        kernel_time_correction_and_finalize<<<blocks_timecorr, threads_timecorr, 0, cudaStream>>>(
            eventOutputGPU.recHitsEB.amplitude.get(),
            eventOutputGPU.recHitsEE.amplitude.get(),
            eventInputGPU.ebDigis.data.get(),
            eventInputGPU.ebDigis.ids.get(),
            eventInputGPU.eeDigis.data.get(),
            eventInputGPU.eeDigis.ids.get(),
            conditions.timeBiasCorrections.EBTimeCorrAmplitudeBins,
            conditions.timeBiasCorrections.EETimeCorrAmplitudeBins,
            conditions.timeBiasCorrections.EBTimeCorrShiftBins,
            conditions.timeBiasCorrections.EETimeCorrShiftBins,
            scratch.timeMax.get(),
            scratch.timeError.get(),
            conditions.pedestals.rms_x12,
            conditions.timeCalibConstants.values,
            eventOutputGPU.recHitsEB.jitter.get(),
            eventOutputGPU.recHitsEE.jitter.get(),
            eventOutputGPU.recHitsEB.jitterError.get(),
            eventOutputGPU.recHitsEE.jitterError.get(),
            eventOutputGPU.recHitsEB.flags.get(),
            eventOutputGPU.recHitsEE.flags.get(),
            conditions.timeBiasCorrections.EBTimeCorrAmplitudeBinsSize,
            conditions.timeBiasCorrections.EETimeCorrAmplitudeBinsSize,
            configParameters.timeConstantTermEB,
            configParameters.timeConstantTermEE,
            conditions.timeOffsetConstant.getEBValue(),
            conditions.timeOffsetConstant.getEEValue(),
            configParameters.timeNconstEB,
            configParameters.timeNconstEE,
            configParameters.amplitudeThreshEB,
            configParameters.amplitudeThreshEE,
            configParameters.outOfTimeThreshG12pEB,
            configParameters.outOfTimeThreshG12pEE,
            configParameters.outOfTimeThreshG12mEB,
            configParameters.outOfTimeThreshG12mEE,
            configParameters.outOfTimeThreshG61pEB,
            configParameters.outOfTimeThreshG61pEE,
            configParameters.outOfTimeThreshG61mEB,
            configParameters.outOfTimeThreshG61mEE,
            offsetForHashes,
            offsetForInputs,
            totalChannels);
        cudaCheck(cudaGetLastError());
      }
    }

  }  // namespace multifit
}  // namespace ecal
