#include <iostream>
#include <limits>
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "AmplitudeComputationCommonKernels.h"
#include "AmplitudeComputationKernels.h"
#include "EcalUncalibRecHitMultiFitAlgoPortable.h"
#include "TimeComputationKernels.h"

//#define DEBUG
//#define ECAL_RECO_ALPAKA_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  using namespace cms::alpakatools;

  void launchKernels(Queue& queue,
                     InputProduct const& digisDevEB,
                     InputProduct const& digisDevEE,
                     OutputProduct& uncalibRecHitsDevEB,
                     OutputProduct& uncalibRecHitsDevEE,
                     EcalMultifitConditionsDevice const& conditionsDev,
                     EcalMultifitParameters const* paramsDev,
                     ConfigurationParameters const& configParams) {
    using digis_type = std::vector<uint16_t>;
    using dids_type = std::vector<uint32_t>;
    // according to the cpu setup  //----> hardcoded
    bool constexpr gainSwitchUseMaxSampleEB = true;
    // according to the cpu setup  //----> hardcoded
    bool constexpr gainSwitchUseMaxSampleEE = false;
    auto constexpr kMaxSamples = EcalDataFrame::MAXSAMPLES;

    auto const ebSize = static_cast<uint32_t>(uncalibRecHitsDevEB.const_view().metadata().size());
    auto const totalChannels = ebSize + static_cast<uint32_t>(uncalibRecHitsDevEE.const_view().metadata().size());

    EventDataForScratchDevice scratch(configParams, totalChannels, queue);

    //
    // 1d preparation kernel
    //
    uint32_t constexpr nchannels_per_block = 32;
    auto constexpr threads_1d = kMaxSamples * nchannels_per_block;
    auto const blocks_1d = cms::alpakatools::divide_up_by(totalChannels * kMaxSamples, threads_1d);
    auto workDivPrep1D = cms::alpakatools::make_workdiv<Acc1D>(blocks_1d, threads_1d);
    // Since the ::ecal::multifit::X objects are non-dynamic Eigen::Matrix types the returned pointers from the buffers
    // and the ::ecal::multifit::X* both point to the data.
    alpaka::exec<Acc1D>(queue,
                        workDivPrep1D,
                        Kernel_prep_1d_and_initialize{},
                        digisDevEB.const_view(),
                        digisDevEE.const_view(),
                        uncalibRecHitsDevEB.view(),
                        uncalibRecHitsDevEE.view(),
                        conditionsDev.const_view(),
                        reinterpret_cast<::ecal::multifit::SampleVector*>(scratch.samplesDevBuf.data()),
                        reinterpret_cast<::ecal::multifit::SampleGainVector*>(scratch.gainsNoiseDevBuf.data()),
                        scratch.hasSwitchToGain6DevBuf.data(),
                        scratch.hasSwitchToGain1DevBuf.data(),
                        scratch.isSaturatedDevBuf.data(),
                        scratch.acStateDevBuf.data(),
                        reinterpret_cast<::ecal::multifit::BXVectorType*>(scratch.activeBXsDevBuf.data()),
                        gainSwitchUseMaxSampleEB,
                        gainSwitchUseMaxSampleEE);

    //
    // 2d preparation kernel
    //
    Vec2D const blocks_2d{1u, totalChannels};  // {y, x} coordiantes
    Vec2D const threads_2d{kMaxSamples, kMaxSamples};
    auto workDivPrep2D = cms::alpakatools::make_workdiv<Acc2D>(blocks_2d, threads_2d);
    alpaka::exec<Acc2D>(queue,
                        workDivPrep2D,
                        Kernel_prep_2d{},
                        digisDevEB.const_view(),
                        digisDevEE.const_view(),
                        conditionsDev.const_view(),
                        reinterpret_cast<::ecal::multifit::SampleGainVector*>(scratch.gainsNoiseDevBuf.data()),
                        reinterpret_cast<::ecal::multifit::SampleMatrix*>(scratch.noisecovDevBuf.data()),
                        reinterpret_cast<::ecal::multifit::PulseMatrixType*>(scratch.pulse_matrixDevBuf.data()),
                        scratch.hasSwitchToGain6DevBuf.data(),
                        scratch.hasSwitchToGain1DevBuf.data(),
                        scratch.isSaturatedDevBuf.data());

    // run minimization kernels
    minimization_procedure(queue,
                           digisDevEB,
                           digisDevEE,
                           uncalibRecHitsDevEB,
                           uncalibRecHitsDevEE,
                           scratch,
                           conditionsDev,
                           configParams,
                           totalChannels);

    if (configParams.shouldRunTimingComputation) {
      //
      // TODO: this guy can run concurrently with other kernels,
      // there is no dependence on the order of execution
      //
      auto const blocks_time_init = blocks_1d;
      auto const threads_time_init = threads_1d;
      auto workDivTimeCompInit1D = cms::alpakatools::make_workdiv<Acc1D>(blocks_time_init, threads_time_init);
      alpaka::exec<Acc1D>(queue,
                          workDivTimeCompInit1D,
                          Kernel_time_computation_init{},
                          digisDevEB.const_view(),
                          digisDevEE.const_view(),
                          conditionsDev.const_view(),
                          scratch.sample_valuesDevBuf.value().data(),
                          scratch.sample_value_errorsDevBuf.value().data(),
                          scratch.ampMaxErrorDevBuf.value().data(),
                          scratch.useless_sample_valuesDevBuf.value().data(),
                          scratch.pedestal_numsDevBuf.value().data());

      //
      // TODO: small kernel only for EB. It needs to be checked if
      /// fusing such small kernels is beneficial in here
      //
      if (ebSize > 0) {
        // we are running only over EB digis
        // therefore we need to create threads/blocks only for that
        auto const threadsFixMGPA = threads_1d;
        auto const blocksFixMGPA = cms::alpakatools::divide_up_by(kMaxSamples * ebSize, threadsFixMGPA);
        auto workDivTimeFixMGPAslew1D = cms::alpakatools::make_workdiv<Acc1D>(blocksFixMGPA, threadsFixMGPA);
        alpaka::exec<Acc1D>(queue,
                            workDivTimeFixMGPAslew1D,
                            Kernel_time_compute_fixMGPAslew{},
                            digisDevEB.const_view(),
                            digisDevEE.const_view(),
                            conditionsDev.const_view(),
                            scratch.sample_valuesDevBuf.value().data(),
                            scratch.sample_value_errorsDevBuf.value().data(),
                            scratch.useless_sample_valuesDevBuf.value().data());
      }

      auto const threads_nullhypot = threads_1d;
      auto const blocks_nullhypot = blocks_1d;
      auto workDivTimeNullhypot1D = cms::alpakatools::make_workdiv<Acc1D>(blocks_nullhypot, threads_nullhypot);
      alpaka::exec<Acc1D>(queue,
                          workDivTimeNullhypot1D,
                          Kernel_time_compute_nullhypot{},
                          scratch.sample_valuesDevBuf.value().data(),
                          scratch.sample_value_errorsDevBuf.value().data(),
                          scratch.useless_sample_valuesDevBuf.value().data(),
                          scratch.chi2sNullHypotDevBuf.value().data(),
                          scratch.sum0sNullHypotDevBuf.value().data(),
                          scratch.sumAAsNullHypotDevBuf.value().data(),
                          totalChannels);

      constexpr uint32_t nchannels_per_block_makeratio = kMaxSamples;
      constexpr auto nthreads_per_channel =
          nchannels_per_block_makeratio * (nchannels_per_block_makeratio - 1) / 2;  // n(n-1)/2
      constexpr auto threads_makeratio = nthreads_per_channel * nchannels_per_block_makeratio;
      auto const blocks_makeratio =
          cms::alpakatools::divide_up_by(nthreads_per_channel * totalChannels, threads_makeratio);
      auto workDivTimeMakeRatio1D = cms::alpakatools::make_workdiv<Acc1D>(blocks_makeratio, threads_makeratio);
      alpaka::exec<Acc1D>(queue,
                          workDivTimeMakeRatio1D,
                          Kernel_time_compute_makeratio{},
                          digisDevEB.const_view(),
                          digisDevEE.const_view(),
                          scratch.sample_valuesDevBuf.value().data(),
                          scratch.sample_value_errorsDevBuf.value().data(),
                          scratch.useless_sample_valuesDevBuf.value().data(),
                          scratch.pedestal_numsDevBuf.value().data(),
                          scratch.sumAAsNullHypotDevBuf.value().data(),
                          scratch.sum0sNullHypotDevBuf.value().data(),
                          scratch.tMaxAlphaBetasDevBuf.value().data(),
                          scratch.tMaxErrorAlphaBetasDevBuf.value().data(),
                          scratch.accTimeMaxDevBuf.value().data(),
                          scratch.accTimeWgtDevBuf.value().data(),
                          scratch.tcStateDevBuf.value().data(),
                          paramsDev,
                          configParams.timeFitLimitsFirstEB,
                          configParams.timeFitLimitsFirstEE,
                          configParams.timeFitLimitsSecondEB,
                          configParams.timeFitLimitsSecondEE);

      auto const threads_findamplchi2 = threads_1d;
      auto const blocks_findamplchi2 = blocks_1d;
      auto workDivTimeFindAmplChi21D = cms::alpakatools::make_workdiv<Acc1D>(blocks_findamplchi2, threads_findamplchi2);
      alpaka::exec<Acc1D>(queue,
                          workDivTimeFindAmplChi21D,
                          Kernel_time_compute_findamplchi2_and_finish{},
                          digisDevEB.const_view(),
                          digisDevEE.const_view(),
                          scratch.sample_valuesDevBuf.value().data(),
                          scratch.sample_value_errorsDevBuf.value().data(),
                          scratch.useless_sample_valuesDevBuf.value().data(),
                          scratch.tMaxAlphaBetasDevBuf.value().data(),
                          scratch.tMaxErrorAlphaBetasDevBuf.value().data(),
                          scratch.accTimeMaxDevBuf.value().data(),
                          scratch.accTimeWgtDevBuf.value().data(),
                          scratch.sumAAsNullHypotDevBuf.value().data(),
                          scratch.sum0sNullHypotDevBuf.value().data(),
                          scratch.chi2sNullHypotDevBuf.value().data(),
                          scratch.tcStateDevBuf.value().data(),
                          scratch.ampMaxAlphaBetaDevBuf.value().data(),
                          scratch.ampMaxErrorDevBuf.value().data(),
                          scratch.timeMaxDevBuf.value().data(),
                          scratch.timeErrorDevBuf.value().data(),
                          paramsDev);

      auto const threads_timecorr = 32;
      auto const blocks_timecorr = cms::alpakatools::divide_up_by(totalChannels, threads_timecorr);
      auto workDivCorrFinal1D = cms::alpakatools::make_workdiv<Acc1D>(blocks_timecorr, threads_timecorr);
      alpaka::exec<Acc1D>(queue,
                          workDivCorrFinal1D,
                          Kernel_time_correction_and_finalize{},
                          digisDevEB.const_view(),
                          digisDevEE.const_view(),
                          uncalibRecHitsDevEB.view(),
                          uncalibRecHitsDevEE.view(),
                          conditionsDev.const_view(),
                          scratch.timeMaxDevBuf.value().data(),
                          scratch.timeErrorDevBuf.value().data(),
                          configParams.timeConstantTermEB,
                          configParams.timeConstantTermEE,
                          configParams.timeNconstEB,
                          configParams.timeNconstEE,
                          configParams.amplitudeThreshEB,
                          configParams.amplitudeThreshEE,
                          configParams.outOfTimeThreshG12pEB,
                          configParams.outOfTimeThreshG12pEE,
                          configParams.outOfTimeThreshG12mEB,
                          configParams.outOfTimeThreshG12mEE,
                          configParams.outOfTimeThreshG61pEB,
                          configParams.outOfTimeThreshG61pEE,
                          configParams.outOfTimeThreshG61mEB,
                          configParams.outOfTimeThreshG61mEE);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit
