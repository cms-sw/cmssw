#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_AmplitudeComputationCommonKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_AmplitudeComputationCommonKernels_h

#include <cstdlib>
#include <limits>
#include <alpaka/alpaka.hpp>

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EigenMatrixTypes_gpu.h"

#include "DeclsForKernels.h"
#include "KernelHelpers.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit {

  ///
  /// assume kernel launch configuration is
  /// (MAXSAMPLES * nchannels, blocks)
  /// TODO: is there a point to split this kernel further to separate reductions
  ///
  class Kernel_prep_1d_and_initialize {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  EcalUncalibratedRecHitDeviceCollection::View uncalibRecHitsEB,
                                  EcalUncalibratedRecHitDeviceCollection::View uncalibRecHitsEE,
                                  EcalMultifitConditionsDevice::ConstView conditionsDev,
                                  ::ecal::multifit::SampleVector* amplitudes,
                                  ::ecal::multifit::SampleGainVector* gainsNoise,
                                  bool* hasSwitchToGain6,
                                  bool* hasSwitchToGain1,
                                  bool* isSaturated,
                                  char* acState,
                                  ::ecal::multifit::BXVectorType* bxs,
                                  bool const gainSwitchUseMaxSampleEB,
                                  bool const gainSwitchUseMaxSampleEE) const {
      constexpr bool dynamicPedestal = false;  //---- default to false, ok
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;
      constexpr int sample_max = 5;
      constexpr int full_pulse_max = 9;
      auto const offsetForHashes = conditionsDev.offsetEE();

      auto const nchannelsEB = digisDevEB.size();
      auto const nchannelsEE = digisDevEE.size();
      auto const nchannels = nchannelsEB + nchannelsEE;
      auto const totalElements = nchannels * nsamples;

      auto const elemsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];

      char* shared_mem = alpaka::getDynSharedMem<char>(acc);
      auto* shr_hasSwitchToGain6 = reinterpret_cast<bool*>(shared_mem);
      auto* shr_hasSwitchToGain1 = shr_hasSwitchToGain6 + elemsPerBlock;
      auto* shr_hasSwitchToGain0 = shr_hasSwitchToGain1 + elemsPerBlock;
      auto* shr_isSaturated = shr_hasSwitchToGain0 + elemsPerBlock;
      auto* shr_hasSwitchToGain0_tmp = shr_isSaturated + elemsPerBlock;
      auto* shr_counts = reinterpret_cast<char*>(shr_hasSwitchToGain0_tmp) + elemsPerBlock;

      for (auto block : cms::alpakatools::uniform_groups(acc, totalElements)) {
        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          // set the output collection size scalars
          if (idx.global == 0) {
            uncalibRecHitsEB.size() = nchannelsEB;
            uncalibRecHitsEE.size() = nchannelsEE;
          }

          auto const ch = idx.global / nsamples;
          // for accessing input arrays
          int const inputTx = ch >= nchannelsEB ? idx.global - nchannelsEB * nsamples : idx.global;
          // eb is first and then ee
          auto const* digis_in = ch >= nchannelsEB ? digisDevEE.data()->data() : digisDevEB.data()->data();
          auto const gainId = ecalMGPA::gainId(digis_in[inputTx]);

          // store into shared mem for initialization
          shr_hasSwitchToGain6[idx.local] = gainId == EcalMgpaBitwiseGain6;
          shr_hasSwitchToGain1[idx.local] = gainId == EcalMgpaBitwiseGain1;
          shr_hasSwitchToGain0_tmp[idx.local] = gainId == EcalMgpaBitwiseGain0;
          shr_hasSwitchToGain0[idx.local] = shr_hasSwitchToGain0_tmp[idx.local];
          shr_counts[idx.local] = 0;
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const sample = idx.local % nsamples;

          // non-divergent branch (except for the last 4 threads)
          if (idx.local <= elemsPerBlock - 5) {
            CMS_UNROLL_LOOP
            for (int i = 0; i < 5; ++i)
              shr_counts[idx.local] += shr_hasSwitchToGain0[idx.local + i];
          }
          shr_isSaturated[idx.local] = shr_counts[idx.local] == 5;

          //
          // unrolled reductions
          //
          if (sample < 5) {
            shr_hasSwitchToGain6[idx.local] = shr_hasSwitchToGain6[idx.local] || shr_hasSwitchToGain6[idx.local + 5];
            shr_hasSwitchToGain1[idx.local] = shr_hasSwitchToGain1[idx.local] || shr_hasSwitchToGain1[idx.local + 5];

            // duplication of hasSwitchToGain0 in order not to
            // introduce another syncthreads
            shr_hasSwitchToGain0_tmp[idx.local] =
                shr_hasSwitchToGain0_tmp[idx.local] || shr_hasSwitchToGain0_tmp[idx.local + 5];
          }
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const sample = idx.local % nsamples;

          if (sample < 2) {
            // note, both threads per channel take value [3] twice to avoid another if
            shr_hasSwitchToGain6[idx.local] = shr_hasSwitchToGain6[idx.local] || shr_hasSwitchToGain6[idx.local + 2] ||
                                              shr_hasSwitchToGain6[idx.local + 3];
            shr_hasSwitchToGain1[idx.local] = shr_hasSwitchToGain1[idx.local] || shr_hasSwitchToGain1[idx.local + 2] ||
                                              shr_hasSwitchToGain1[idx.local + 3];

            shr_hasSwitchToGain0_tmp[idx.local] = shr_hasSwitchToGain0_tmp[idx.local] ||
                                                  shr_hasSwitchToGain0_tmp[idx.local + 2] ||
                                                  shr_hasSwitchToGain0_tmp[idx.local + 3];

            // sample < 2 -> first 2 threads of each channel will be used here
            // => 0 -> will compare 3 and 4 and put into 0
            // => 1 -> will compare 4 and 5 and put into 1
            shr_isSaturated[idx.local] = shr_isSaturated[idx.local + 3] || shr_isSaturated[idx.local + 4];
          }
        }

        alpaka::syncBlockThreads(acc);

        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const ch = idx.global / nsamples;
          auto const sample = idx.local % nsamples;

          if (sample == 0) {
            shr_hasSwitchToGain6[idx.local] = shr_hasSwitchToGain6[idx.local] || shr_hasSwitchToGain6[idx.local + 1];
            shr_hasSwitchToGain1[idx.local] = shr_hasSwitchToGain1[idx.local] || shr_hasSwitchToGain1[idx.local + 1];
            shr_hasSwitchToGain0_tmp[idx.local] =
                shr_hasSwitchToGain0_tmp[idx.local] || shr_hasSwitchToGain0_tmp[idx.local + 1];

            hasSwitchToGain6[ch] = shr_hasSwitchToGain6[idx.local];
            hasSwitchToGain1[ch] = shr_hasSwitchToGain1[idx.local];

            shr_isSaturated[idx.local + 3] = shr_isSaturated[idx.local] || shr_isSaturated[idx.local + 1];
            isSaturated[ch] = shr_isSaturated[idx.local + 3];
          }
        }

        // TODO: w/o this sync, there is a race
        // if (idx.local == sample_max) below uses max sample thread, not for 0 sample
        // check if we can remove it
        alpaka::syncBlockThreads(acc);

        for (auto idx : cms::alpakatools::uniform_group_elements(acc, block, totalElements)) {
          auto const ch = idx.global / nsamples;
          auto const sample = idx.local % nsamples;

          // for accessing input arrays
          int const inputCh = ch >= nchannelsEB ? ch - nchannelsEB : ch;
          int const inputTx = ch >= nchannelsEB ? idx.global - nchannelsEB * nsamples : idx.global;

          auto const* dids = ch >= nchannelsEB ? digisDevEE.id() : digisDevEB.id();
          auto const did = DetId{dids[inputCh]};
          auto const isBarrel = did.subdetId() == EcalBarrel;
          // TODO offset for ee, 0 for eb
          auto const hashedId = isBarrel ? reconstruction::hashedIndexEB(did.rawId())
                                         : offsetForHashes + reconstruction::hashedIndexEE(did.rawId());

          // eb is first and then ee
          auto const* digis_in = ch >= nchannelsEB ? digisDevEE.data()->data() : digisDevEB.data()->data();

          auto* amplitudesForMinimization = reinterpret_cast<::ecal::multifit::SampleVector*>(
              ch >= nchannelsEB ? uncalibRecHitsEE.outOfTimeAmplitudes()->data()
                                : uncalibRecHitsEB.outOfTimeAmplitudes()->data());
          auto* energies = ch >= nchannelsEB ? uncalibRecHitsEE.amplitude() : uncalibRecHitsEB.amplitude();
          auto* chi2 = ch >= nchannelsEB ? uncalibRecHitsEE.chi2() : uncalibRecHitsEB.chi2();
          auto* g_pedestal = ch >= nchannelsEB ? uncalibRecHitsEE.pedestal() : uncalibRecHitsEB.pedestal();
          auto* dids_out = ch >= nchannelsEB ? uncalibRecHitsEE.id() : uncalibRecHitsEB.id();
          auto* flags = ch >= nchannelsEB ? uncalibRecHitsEE.flags() : uncalibRecHitsEB.flags();

          auto const adc = ecalMGPA::adc(digis_in[inputTx]);
          auto const gainId = ecalMGPA::gainId(digis_in[inputTx]);
          ::ecal::multifit::SampleVector::Scalar amplitude = 0.;
          ::ecal::multifit::SampleVector::Scalar pedestal = 0.;
          ::ecal::multifit::SampleVector::Scalar gainratio = 0.;

          // TODO: divergent branch
          if (gainId == 0 || gainId == 3) {
            pedestal = conditionsDev.pedestals_mean_x1()[hashedId];
            gainratio = conditionsDev.gain6Over1()[hashedId] * conditionsDev.gain12Over6()[hashedId];
            gainsNoise[ch](sample) = 2;
          } else if (gainId == 1) {
            pedestal = conditionsDev.pedestals_mean_x12()[hashedId];
            gainratio = 1.;
            gainsNoise[ch](sample) = 0;
          } else if (gainId == 2) {
            pedestal = conditionsDev.pedestals_mean_x6()[hashedId];
            gainratio = conditionsDev.gain12Over6()[hashedId];
            gainsNoise[ch](sample) = 1;
          }

          // TODO: compile time constant -> branch should be non-divergent
          if (dynamicPedestal)
            amplitude = static_cast<::ecal::multifit::SampleVector::Scalar>(adc) * gainratio;
          else
            amplitude = (static_cast<::ecal::multifit::SampleVector::Scalar>(adc) - pedestal) * gainratio;
          amplitudes[ch][sample] = amplitude;

#ifdef ECAL_RECO_ALPAKA_DEBUG
          printf("%d %d %d %d %f %f %f\n", idx.global, ch, sample, adc, amplitude, pedestal, gainratio);
          if (adc == 0)
            printf("adc is zero\n");
#endif

          //
          // initialization
          //
          amplitudesForMinimization[inputCh](sample) = 0;
          bxs[ch](sample) = sample - 5;

          // select the thread for the max sample
          //---> hardcoded above to be 5th sample, ok
          if (sample == sample_max) {
            //
            // initialization
            //
            acState[ch] = static_cast<char>(MinimizationState::NotFinished);
            energies[inputCh] = 0;
            chi2[inputCh] = 0;
            g_pedestal[inputCh] = 0;
            uint32_t flag = 0;
            dids_out[inputCh] = did.rawId();

            // start of this channel in shared mem
            auto const chStart = idx.local - sample_max;
            // thread for the max sample in shared mem
            auto const threadMax = idx.local;
            auto const gainSwitchUseMaxSample = isBarrel ? gainSwitchUseMaxSampleEB : gainSwitchUseMaxSampleEE;

            // this flag setting is applied to all of the cases
            if (shr_hasSwitchToGain6[chStart])
              flag |= 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain6;
            if (shr_hasSwitchToGain1[chStart])
              flag |= 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;

            // this corresponds to cpu branching on lastSampleBeforeSaturation
            // likely false
            // check only for the idx.local corresponding to sample==0
            if (sample == 0 && shr_hasSwitchToGain0_tmp[idx.local]) {
              // assign for the case some sample having gainId == 0
              //energies[inputCh] = amplitudes[ch][sample_max];
              energies[inputCh] = amplitude;

              // check if samples before sample_max have true
              bool saturated_before_max = false;
              CMS_UNROLL_LOOP
              for (char ii = 0; ii < 5; ++ii)
                saturated_before_max = saturated_before_max || shr_hasSwitchToGain0[chStart + ii];

              // if saturation is in the max sample and not in the first 5
              if (!saturated_before_max && shr_hasSwitchToGain0[threadMax])
                energies[inputCh] = 49140;  // 4095 * 12 (maximum ADC range * MultiGainPreAmplifier (MGPA) gain)
                                            // This is the actual maximum range that is set when we saturate.
                                            //---- AM FIXME : no pedestal subtraction???
                                            //It should be "(4095. - pedestal) * gainratio"

              // set state flag to terminate further processing of this channel
              acState[ch] = static_cast<char>(MinimizationState::Precomputed);
              flag |= 0x1 << EcalUncalibratedRecHit::kSaturated;
              flags[inputCh] = flag;
              continue;
            }

            // according to cpu version
            //            auto max_amplitude = amplitudes[ch][sample_max];
            auto const max_amplitude = amplitude;
            // pulse shape template value
            auto shape_value = conditionsDev.pulseShapes()[hashedId][full_pulse_max - 7];
            // note, no syncing as the same thread will be accessing here
            bool hasGainSwitch =
                shr_hasSwitchToGain6[chStart] || shr_hasSwitchToGain1[chStart] || shr_isSaturated[chStart + 3];

            // pedestal is final unconditionally
            g_pedestal[inputCh] = pedestal;
            if (hasGainSwitch && gainSwitchUseMaxSample) {
              // thread for sample=0 will access the right guys
              energies[inputCh] = max_amplitude / shape_value;
              acState[ch] = static_cast<char>(MinimizationState::Precomputed);
              flags[inputCh] = flag;
              continue;
            }

            // will be used in the future for setting state
            auto const rmsForChecking = conditionsDev.pedestals_rms_x12()[hashedId];

            // this happens cause sometimes rms_x12 is 0...
            // needs to be checkec why this is the case
            // general case here is that noisecov is a Zero matrix
            if (rmsForChecking == 0) {
              acState[ch] = static_cast<char>(MinimizationState::Precomputed);
              flags[inputCh] = flag;
              continue;
            }

            // for the case when no shortcuts were taken
            flags[inputCh] = flag;
          }
        }
      }
    }
  };

  ///
  /// assume kernel launch configuration is
  /// ([MAXSAMPLES, MAXSAMPLES], nchannels)
  ///
  class Kernel_prep_2d {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  EcalDigiDeviceCollection::ConstView digisDevEB,
                                  EcalDigiDeviceCollection::ConstView digisDevEE,
                                  EcalMultifitConditionsDevice::ConstView conditionsDev,
                                  ::ecal::multifit::SampleGainVector const* gainsNoise,
                                  ::ecal::multifit::SampleMatrix* noisecov,
                                  ::ecal::multifit::PulseMatrixType* pulse_matrix,
                                  bool const* hasSwitchToGain6,
                                  bool const* hasSwitchToGain1,
                                  bool const* isSaturated) const {
      constexpr auto nsamples = EcalDataFrame::MAXSAMPLES;
      auto const offsetForHashes = conditionsDev.offsetEE();
      auto const nchannelsEB = digisDevEB.size();
      constexpr float addPedestalUncertainty = 0.f;
      constexpr bool dynamicPedestal = false;
      constexpr bool simplifiedNoiseModelForGainSwitch = true;  //---- default is true

      // pulse matrix
      auto const* pulse_shapes = reinterpret_cast<const EcalPulseShape*>(conditionsDev.pulseShapes()->data());

      auto const blockDimX = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[1u];
      auto const elemsPerBlockX = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[1u];
      auto const elemsPerBlockY = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
      Vec2D const size_2d = {elemsPerBlockY, blockDimX * elemsPerBlockX};  // {y, x} coordinates

      for (auto ndindex : cms::alpakatools::uniform_elements_nd(acc, size_2d)) {
        auto const ch = ndindex[1] / nsamples;
        auto const tx = ndindex[1] % nsamples;
        auto const ty = ndindex[0];

        // to access input arrays (ids and digis only)
        int const inputCh = ch >= nchannelsEB ? ch - nchannelsEB : ch;
        auto const* dids = ch >= nchannelsEB ? digisDevEE.id() : digisDevEB.id();

        auto const did = DetId{dids[inputCh]};
        auto const isBarrel = did.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());
        auto const* G12SamplesCorrelation = isBarrel ? conditionsDev.sampleCorrelation_EB_G12().data()
                                                     : conditionsDev.sampleCorrelation_EE_G12().data();
        auto const* G6SamplesCorrelation =
            isBarrel ? conditionsDev.sampleCorrelation_EB_G6().data() : conditionsDev.sampleCorrelation_EE_G6().data();
        auto const* G1SamplesCorrelation =
            isBarrel ? conditionsDev.sampleCorrelation_EB_G1().data() : conditionsDev.sampleCorrelation_EE_G1().data();
        auto const hasGainSwitch = hasSwitchToGain6[ch] || hasSwitchToGain1[ch] || isSaturated[ch];

        auto const vidx = std::abs(static_cast<int>(ty) - static_cast<int>(tx));

        // non-divergent branch for all threads per block
        if (hasGainSwitch) {
          // TODO: did not include simplified noise model
          float noise_value = 0;

          // non-divergent branch - all threads per block
          // TODO: all of these constants indicate that
          // that these parts could be splitted into completely different
          // kernels and run one of them only depending on the config
          if (simplifiedNoiseModelForGainSwitch) {
            constexpr int isample_max = 5;  // according to cpu defs
            auto const gainidx = gainsNoise[ch][isample_max];

            // non-divergent branches
            if (gainidx == 0) {
              auto const rms_x12 = conditionsDev.pedestals_rms_x12()[hashedId];
              noise_value = rms_x12 * rms_x12 * G12SamplesCorrelation[vidx];
            } else if (gainidx == 1) {
              auto const gain12Over6 = conditionsDev.gain12Over6()[hashedId];
              auto const rms_x6 = conditionsDev.pedestals_rms_x6()[hashedId];
              noise_value = gain12Over6 * gain12Over6 * rms_x6 * rms_x6 * G6SamplesCorrelation[vidx];
            } else if (gainidx == 2) {
              auto const gain12Over6 = conditionsDev.gain12Over6()[hashedId];
              auto const gain6Over1 = conditionsDev.gain6Over1()[hashedId];
              auto const gain12Over1 = gain12Over6 * gain6Over1;
              auto const rms_x1 = conditionsDev.pedestals_rms_x1()[hashedId];
              noise_value = gain12Over1 * gain12Over1 * rms_x1 * rms_x1 * G1SamplesCorrelation[vidx];
            }
            if (!dynamicPedestal && addPedestalUncertainty > 0.f)
              noise_value += addPedestalUncertainty * addPedestalUncertainty;
          } else {
            int gainidx = 0;
            char mask = gainidx;
            int pedestal = gainsNoise[ch][ty] == mask ? 1 : 0;
            //            NB: gainratio is 1, that is why it does not appear in the formula
            auto const rms_x12 = conditionsDev.pedestals_rms_x12()[hashedId];
            noise_value += rms_x12 * rms_x12 * pedestal * G12SamplesCorrelation[vidx];
            // non-divergent branch
            if (!dynamicPedestal && addPedestalUncertainty > 0.f) {
              noise_value += addPedestalUncertainty * addPedestalUncertainty * pedestal;  // gainratio is 1
            }

            //
            gainidx = 1;
            mask = gainidx;
            pedestal = gainsNoise[ch][ty] == mask ? 1 : 0;
            auto const gain12Over6 = conditionsDev.gain12Over6()[hashedId];
            auto const rms_x6 = conditionsDev.pedestals_rms_x6()[hashedId];
            noise_value += gain12Over6 * gain12Over6 * rms_x6 * rms_x6 * pedestal * G6SamplesCorrelation[vidx];
            // non-divergent branch
            if (!dynamicPedestal && addPedestalUncertainty > 0.f) {
              noise_value += gain12Over6 * gain12Over6 * addPedestalUncertainty * addPedestalUncertainty * pedestal;
            }

            //
            gainidx = 2;
            mask = gainidx;
            pedestal = gainsNoise[ch][ty] == mask ? 1 : 0;
            auto const gain6Over1 = conditionsDev.gain6Over1()[hashedId];
            auto const gain12Over1 = gain12Over6 * gain6Over1;
            auto const rms_x1 = conditionsDev.pedestals_rms_x1()[hashedId];
            noise_value += gain12Over1 * gain12Over1 * rms_x1 * rms_x1 * pedestal * G1SamplesCorrelation[vidx];
            // non-divergent branch
            if (!dynamicPedestal && addPedestalUncertainty > 0.f) {
              noise_value += gain12Over1 * gain12Over1 * addPedestalUncertainty * addPedestalUncertainty * pedestal;
            }
          }

          noisecov[ch](ty, tx) = noise_value;
        } else {
          auto const rms = conditionsDev.pedestals_rms_x12()[hashedId];
          float noise_value = rms * rms * G12SamplesCorrelation[vidx];
          if (!dynamicPedestal && addPedestalUncertainty > 0.f) {
            //----  add fully correlated component to noise covariance to inflate pedestal uncertainty
            noise_value += addPedestalUncertainty * addPedestalUncertainty;
          }
          noisecov[ch](ty, tx) = noise_value;
        }

        auto const posToAccess = 9 - static_cast<int>(tx) + static_cast<int>(ty);  // see cpu for reference
        float const value = posToAccess >= 7 ? pulse_shapes[hashedId].pdfval[posToAccess - 7] : 0;
        pulse_matrix[ch](ty, tx) = value;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit

namespace alpaka::trait {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE;
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::multifit;

  //! The trait for getting the size of the block shared dynamic memory for Kernel_prep_1d_and_initialize.
  template <>
  struct BlockSharedMemDynSizeBytes<Kernel_prep_1d_and_initialize, Acc1D> {
    //! \return The size of the shared memory allocated for a block.
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(Kernel_prep_1d_and_initialize const&,
                                                                 TVec const& threadsPerBlock,
                                                                 TVec const& elemsPerThread,
                                                                 TArgs const&...) -> std::size_t {
      // return the amount of dynamic shared memory needed
      std::size_t bytes = threadsPerBlock[0u] * elemsPerThread[0u] * (5 * sizeof(bool) + sizeof(char));
      return bytes;
    }
  };
}  // namespace alpaka::trait

#endif  // RecoLocalCalo_EcalRecProducers_plugins_AmplitudeComputationCommonKernels_h
