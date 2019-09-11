#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo_gpu_new.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"

#include <iostream>
#include <limits>

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/Common.h"

#include "cuda.h"

#include "AmplitudeComputationCommonKernels.h"
#include "AmplitudeComputationKernelsV1.h"
#include "TimeComputationKernels.h"

//#define DEBUG

//#define ECAL_RECO_CUDA_DEBUG

namespace ecal {
  namespace multifit {

    void entryPoint(EventInputDataCPU const& eventInputCPU,
                    EventInputDataGPU& eventInputGPU,
                    EventOutputDataGPU& eventOutputGPU,
                    EventDataForScratchGPU& scratch,
                    ConditionsProducts const& conditions,
                    ConfigurationParameters const& configParameters,
                    cuda::stream_t<>& cudaStream) {
      using digis_type = std::vector<uint16_t>;
      using dids_type = std::vector<uint32_t>;
      // accodring to the cpu setup  //----> hardcoded
      bool const gainSwitchUseMaxSampleEB = true;
      // accodring to the cpu setup  //----> hardcoded
      bool const gainSwitchUseMaxSampleEE = false;

      uint32_t const offsetForHashes = conditions.offsetForHashes;
      unsigned int totalChannels = eventInputCPU.ebDigis.size() + eventInputCPU.eeDigis.size();

      // temporary for recording
      /*cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaCheck( cudaEventCreate(&start_event) );
    cudaCheck( cudaEventCreate(&end_event) );

    cudaCheck (cudaEventRecord(start_event, 0) );
    */

      //
      // in what follows we copy eb then ee.
      // offset by size
      //

      //
      // copy event data: digis + ids, not really async as vectors have default
      // allocators
      //
      cudaCheck(cudaMemcpyAsync(eventInputGPU.digis,
                                eventInputCPU.ebDigis.data().data(),
                                eventInputCPU.ebDigis.data().size() * sizeof(digis_type::value_type),
                                cudaMemcpyHostToDevice,
                                cudaStream.id()));
      cudaCheck(cudaMemcpyAsync(eventInputGPU.digis + eventInputCPU.ebDigis.data().size(),
                                eventInputCPU.eeDigis.data().data(),
                                eventInputCPU.eeDigis.data().size() * sizeof(digis_type::value_type),
                                cudaMemcpyHostToDevice,
                                cudaStream.id()));

      cudaCheck(cudaMemcpyAsync(eventInputGPU.ids,
                                eventInputCPU.ebDigis.ids().data(),
                                eventInputCPU.ebDigis.ids().size() * sizeof(dids_type::value_type),
                                cudaMemcpyHostToDevice,
                                cudaStream.id()));
      cudaCheck(cudaMemcpyAsync(eventInputGPU.ids + eventInputCPU.ebDigis.ids().size(),
                                eventInputCPU.eeDigis.ids().data(),
                                eventInputCPU.eeDigis.ids().size() * sizeof(dids_type::value_type),
                                cudaMemcpyHostToDevice,
                                cudaStream.id()));

      //
      // 1d preparation kernel
      //
      unsigned int nchannels_per_block = 32;
      unsigned int threads_1d = 10 * nchannels_per_block;
      unsigned int blocks_1d = threads_1d > 10 * totalChannels ? 1 : (totalChannels * 10 + threads_1d - 1) / threads_1d;
      int shared_bytes = nchannels_per_block * EcalDataFrame::MAXSAMPLES *
                         (sizeof(bool) + sizeof(bool) + sizeof(bool) + sizeof(bool) + sizeof(char) + sizeof(bool));
      kernel_prep_1d_and_initialize<<<blocks_1d, threads_1d, shared_bytes, cudaStream.id()>>>(
          conditions.pulseShapes.values,
          eventInputGPU.digis,
          eventInputGPU.ids,
          scratch.samples,
          (SampleVector*)eventOutputGPU.amplitudesAll,
          scratch.gainsNoise,
          conditions.pedestals.mean_x1,
          conditions.pedestals.mean_x12,
          conditions.pedestals.rms_x12,
          conditions.pedestals.mean_x6,
          conditions.gainRatios.gain6Over1,
          conditions.gainRatios.gain12Over6,
          scratch.hasSwitchToGain6,
          scratch.hasSwitchToGain1,
          scratch.isSaturated,
          eventOutputGPU.amplitude,
          eventOutputGPU.chi2,
          eventOutputGPU.pedestal,
          eventOutputGPU.flags,
          scratch.acState,
          scratch.activeBXs,
          offsetForHashes,
          gainSwitchUseMaxSampleEB,
          gainSwitchUseMaxSampleEE,
          totalChannels);
      cudaCheck(cudaGetLastError());

      //
      // 2d preparation kernel
      //
      int blocks_2d = totalChannels;
      dim3 threads_2d{10, 10};
      kernel_prep_2d<<<blocks_2d, threads_2d, 0, cudaStream.id()>>>(
          conditions.pulseCovariances.values,
          scratch.pulse_covariances,
          scratch.gainsNoise,
          eventInputGPU.ids,
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
          scratch.noisecov,
          scratch.pulse_matrix,
          conditions.pulseShapes.values,
          scratch.hasSwitchToGain6,
          scratch.hasSwitchToGain1,
          scratch.isSaturated,
          offsetForHashes);
      cudaCheck(cudaGetLastError());

      // run minimization kernels
      v1::minimization_procedure(
          eventInputCPU, eventInputGPU, eventOutputGPU, scratch, conditions, configParameters, cudaStream);

      if (configParameters.shouldRunTimingComputation) {
        //
        // TODO: this guy can run concurrently with other kernels,
        // there is no dependence on the order of execution
        //
        unsigned int threads_time_init = threads_1d;
        unsigned int blocks_time_init = blocks_1d;
        int sharedBytesInit = 2 * threads_time_init * sizeof(SampleVector::Scalar);
        kernel_time_computation_init<<<blocks_time_init, threads_time_init, sharedBytesInit, cudaStream.id()>>>(
            eventInputGPU.digis,
            eventInputGPU.ids,
            conditions.pedestals.rms_x12,
            conditions.pedestals.rms_x6,
            conditions.pedestals.rms_x1,
            conditions.pedestals.mean_x12,
            conditions.pedestals.mean_x6,
            conditions.pedestals.mean_x1,
            conditions.gainRatios.gain12Over6,
            conditions.gainRatios.gain6Over1,
            scratch.sample_values,
            scratch.sample_value_errors,
            scratch.ampMaxError,
            scratch.useless_sample_values,
            scratch.pedestal_nums,
            offsetForHashes,
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
            threadsFixMGPA > 10 * eventInputCPU.ebDigis.size()
                ? 1
                : (10 * eventInputCPU.ebDigis.size() + threadsFixMGPA - 1) / threadsFixMGPA;
        kernel_time_compute_fixMGPAslew<<<blocksFixMGPA, threadsFixMGPA, 0, cudaStream.id()>>>(
            eventInputGPU.digis,
            scratch.sample_values,
            scratch.sample_value_errors,
            scratch.useless_sample_values,
            conditions.sampleMask.getEcalSampleMaskRecordEB(),
            totalChannels);
        cudaCheck(cudaGetLastError());

        //
        //
        //
        int sharedBytes = EcalDataFrame::MAXSAMPLES * nchannels_per_block * 4 * sizeof(SampleVector::Scalar);
        auto const threads_nullhypot = threads_1d;
        auto const blocks_nullhypot = blocks_1d;
        kernel_time_compute_nullhypot<<<blocks_nullhypot, threads_nullhypot, sharedBytes, cudaStream.id()>>>(
            scratch.sample_values,
            scratch.sample_value_errors,
            scratch.useless_sample_values,
            scratch.chi2sNullHypot,
            scratch.sum0sNullHypot,
            scratch.sumAAsNullHypot,
            totalChannels);
        cudaCheck(cudaGetLastError());

        unsigned int nchannels_per_block_makeratio = 10;
        unsigned int threads_makeratio = 45 * nchannels_per_block_makeratio;
        unsigned int blocks_makeratio = threads_makeratio > 45 * totalChannels
                                            ? 1
                                            : (totalChannels * 45 + threads_makeratio - 1) / threads_makeratio;
        int sharedBytesMakeRatio = 5 * threads_makeratio * sizeof(SampleVector::Scalar);
        kernel_time_compute_makeratio<<<blocks_makeratio, threads_makeratio, sharedBytesMakeRatio, cudaStream.id()>>>(
            scratch.sample_values,
            scratch.sample_value_errors,
            eventInputGPU.ids,
            scratch.useless_sample_values,
            scratch.pedestal_nums,
            configParameters.amplitudeFitParametersEB,
            configParameters.amplitudeFitParametersEE,
            configParameters.timeFitParametersEB,
            configParameters.timeFitParametersEE,
            scratch.sumAAsNullHypot,
            scratch.sum0sNullHypot,
            scratch.tMaxAlphaBetas,
            scratch.tMaxErrorAlphaBetas,
            scratch.accTimeMax,
            scratch.accTimeWgt,
            scratch.tcState,
            configParameters.timeFitParametersSizeEB,
            configParameters.timeFitParametersSizeEE,
            configParameters.timeFitLimitsFirstEB,
            configParameters.timeFitLimitsFirstEE,
            configParameters.timeFitLimitsSecondEB,
            configParameters.timeFitLimitsSecondEE,
            totalChannels);
        cudaCheck(cudaGetLastError());

        //
        //
        //
        auto const threads_findamplchi2 = threads_1d;
        auto const blocks_findamplchi2 = blocks_1d;
        int const sharedBytesFindAmplChi2 = 2 * threads_findamplchi2 * sizeof(SampleVector::Scalar);
        kernel_time_compute_findamplchi2_and_finish<<<blocks_findamplchi2,
                                                      threads_findamplchi2,
                                                      sharedBytesFindAmplChi2,
                                                      cudaStream.id()>>>(scratch.sample_values,
                                                                         scratch.sample_value_errors,
                                                                         eventInputGPU.ids,
                                                                         scratch.useless_sample_values,
                                                                         scratch.tMaxAlphaBetas,
                                                                         scratch.tMaxErrorAlphaBetas,
                                                                         scratch.accTimeMax,
                                                                         scratch.accTimeWgt,
                                                                         configParameters.amplitudeFitParametersEB,
                                                                         configParameters.amplitudeFitParametersEE,
                                                                         scratch.sumAAsNullHypot,
                                                                         scratch.sum0sNullHypot,
                                                                         scratch.chi2sNullHypot,
                                                                         scratch.tcState,
                                                                         scratch.ampMaxAlphaBeta,
                                                                         scratch.ampMaxError,
                                                                         scratch.timeMax,
                                                                         scratch.timeError,
                                                                         totalChannels);
        cudaCheck(cudaGetLastError());

        //
        //
        //
        auto const threads_timecorr = 32;
        auto const blocks_timecorr =
            threads_timecorr > totalChannels ? 1 : (totalChannels + threads_timecorr - 1) / threads_timecorr;
        kernel_time_correction_and_finalize<<<blocks_timecorr, threads_timecorr, 0, cudaStream.id()>>>(
            eventOutputGPU.amplitude,
            eventInputGPU.digis,
            eventInputGPU.ids,
            conditions.timeBiasCorrections.EBTimeCorrAmplitudeBins,
            conditions.timeBiasCorrections.EETimeCorrAmplitudeBins,
            conditions.timeBiasCorrections.EBTimeCorrShiftBins,
            conditions.timeBiasCorrections.EETimeCorrShiftBins,
            scratch.timeMax,
            scratch.timeError,
            conditions.pedestals.rms_x12,
            conditions.timeCalibConstants.values,
            eventOutputGPU.jitter,
            eventOutputGPU.jitterError,
            eventOutputGPU.flags,
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
            totalChannels);
        cudaCheck(cudaGetLastError());
      }

      /*
    cudaEventRecord(end_event, 0);
    cudaEventSynchronize(end_event);
    float ms;
    cudaEventElapsedTime(&ms, start_event, end_event);
    std::cout << "elapsed time = " << ms << std::endl;
    */
    }

  }  // namespace multifit
}  // namespace ecal
