#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EnergyComputationKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EnergyComputationKernels_h

#include <alpaka/alpaka.hpp>

#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitParametersDevice.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "DeclsForKernels.h"
#include "KernelHelpers.h"

//#define DEBUG
//#define ECAL_RECO_ALPAKA_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit {

  ALPAKA_STATIC_ACC_MEM_CONSTANT constexpr float ip10[] = {
      1.e5f, 1.e4f, 1.e3f, 1.e2f, 1.e1f, 1.e0f, 1.e-1f, 1.e-2f, 1.e-3f, 1.e-4};

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkUncalibRecHitFlag(uint32_t const flags, EcalUncalibratedRecHit::Flags flag) {
    return flags & (0x1 << flag);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void setFlag(uint32_t& flags, EcalRecHit::Flags flag) { flags |= (0x1 << flag); }

  ALPAKA_FN_ACC void makeRecHit(int const inputCh,
                                uint32_t const* didCh,
                                float const* amplitude,
                                float const* amplitudeError,
                                float const* jitter,
                                uint32_t const* aux,
                                float const* chi2_in,
                                uint32_t const* flags_in,
                                uint32_t* did,
                                float* energy,
                                float* time,
                                uint32_t* flagBits,
                                uint32_t* extra,
                                EcalRecHitConditionsDevice::ConstView conditionsDev,
                                EcalRecHitParametersDevice::Product const* parametersDev,
                                // time, used for time dependent corrections
                                edm::TimeValue_t const& eventTime,
                                // configuration
                                bool const isPhase2,
                                bool const killDeadChannels,
                                bool const recoverIsolatedChannels,
                                bool const recoverVFE,
                                bool const recoverFE,
                                float const laserMIN,
                                float const laserMAX,
                                uint32_t flagmask);

  class KernelCreateEcalRechitPhase2 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  EcalUncalibratedRecHitDeviceCollection::ConstView uncalibRecHits,
                                  EcalRecHitDeviceCollection::View recHits,
                                  EcalRecHitConditionsDevice::ConstView conditionsDev,
                                  EcalRecHitParametersDevice::Product const* parametersDev,
                                  // time, used for time dependent corrections
                                  edm::TimeValue_t const& eventTime,
                                  ConfigurationParameters const& configParams) const {
      auto const nchannels = uncalibRecHits.size();

      for (auto ch : cms::alpakatools::uniform_elements(acc, nchannels)) {
        // set the output collection size scalar
        if (ch == 0) {
          recHits.size() = nchannels;
        }

        makeRecHit(ch,
                   uncalibRecHits.id(),
                   uncalibRecHits.amplitude(),
                   uncalibRecHits.amplitudeError(),
                   uncalibRecHits.jitter(),
                   uncalibRecHits.aux(),
                   uncalibRecHits.chi2(),
                   uncalibRecHits.flags(),
                   recHits.id(),
                   recHits.energy(),
                   recHits.time(),
                   recHits.flagBits(),
                   recHits.extra(),
                   conditionsDev,
                   parametersDev,
                   eventTime,
                   true,
                   configParams.killDeadChannels,
                   configParams.recoverEBIsolatedChannels,
                   configParams.recoverEBVFE,
                   configParams.recoverEBFE,
                   configParams.EBLaserMIN,
                   configParams.EBLaserMAX,
                   configParams.flagmask);

      }  // end channel
    }
  };

  class KernelCreateEcalRechit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  EcalUncalibratedRecHitDeviceCollection::ConstView ebUncalibRecHits,
                                  EcalUncalibratedRecHitDeviceCollection::ConstView eeUncalibRecHits,
                                  EcalRecHitDeviceCollection::View ebRecHits,
                                  EcalRecHitDeviceCollection::View eeRecHits,
                                  EcalRecHitConditionsDevice::ConstView conditionsDev,
                                  EcalRecHitParametersDevice::Product const* parametersDev,
                                  // time, used for time dependent corrections
                                  edm::TimeValue_t const& eventTime,
                                  ConfigurationParameters const& configParams) const {
      auto const nchannelsEB = ebUncalibRecHits.size();
      auto const nchannelsEE = eeUncalibRecHits.size();
      auto const nchannels = nchannelsEB + nchannelsEE;

      for (auto ch : cms::alpakatools::uniform_elements(acc, nchannels)) {
        // set the output collection size scalars
        if (ch == 0) {
          ebRecHits.size() = nchannelsEB;
        } else if (ch == nchannelsEB) {
          eeRecHits.size() = nchannelsEE;
        }

        bool const isEndcap = ch >= nchannelsEB;
        int const inputCh = isEndcap ? ch - nchannelsEB : ch;

        // inputs
        auto const* didCh = isEndcap ? eeUncalibRecHits.id() : ebUncalibRecHits.id();
        auto const* amplitude = isEndcap ? eeUncalibRecHits.amplitude() : ebUncalibRecHits.amplitude();
        auto const* amplitudeError = isEndcap ? eeUncalibRecHits.amplitudeError() : ebUncalibRecHits.amplitudeError();
        auto const* jitter = isEndcap ? eeUncalibRecHits.jitter() : ebUncalibRecHits.jitter();
        auto const* aux = isEndcap ? eeUncalibRecHits.aux() : ebUncalibRecHits.aux();
        auto const* chi2_in = isEndcap ? eeUncalibRecHits.chi2() : ebUncalibRecHits.chi2();
        auto const* flags_in = isEndcap ? eeUncalibRecHits.flags() : ebUncalibRecHits.flags();

        // outputs
        auto* did = isEndcap ? eeRecHits.id() : ebRecHits.id();
        auto* energy = isEndcap ? eeRecHits.energy() : ebRecHits.energy();
        auto* time = isEndcap ? eeRecHits.time() : ebRecHits.time();
        auto* flagBits = isEndcap ? eeRecHits.flagBits() : ebRecHits.flagBits();
        auto* extra = isEndcap ? eeRecHits.extra() : ebRecHits.extra();

        bool const recoverIsolatedChannels =
            isEndcap ? configParams.recoverEEIsolatedChannels : configParams.recoverEBIsolatedChannels;
        bool const recoverVFE = isEndcap ? configParams.recoverEEVFE : configParams.recoverEBVFE;
        bool const recoverFE = isEndcap ? configParams.recoverEEFE : configParams.recoverEBFE;
        float const laserMIN = isEndcap ? configParams.EELaserMIN : configParams.EBLaserMIN;
        float const laserMAX = isEndcap ? configParams.EELaserMAX : configParams.EBLaserMAX;

        makeRecHit(inputCh,
                   didCh,
                   amplitude,
                   amplitudeError,
                   jitter,
                   aux,
                   chi2_in,
                   flags_in,
                   did,
                   energy,
                   time,
                   flagBits,
                   extra,
                   conditionsDev,
                   parametersDev,
                   eventTime,
                   false,
                   configParams.killDeadChannels,
                   recoverIsolatedChannels,
                   recoverVFE,
                   recoverFE,
                   laserMIN,
                   laserMAX,
                   configParams.flagmask);

      }  // end channel
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit

#endif  // RecoLocalCalo_EcalRecProducers_plugins_alpaka_EnergyComputationKernels_h
