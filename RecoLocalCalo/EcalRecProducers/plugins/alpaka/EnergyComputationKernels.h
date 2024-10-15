#ifndef RecoLocalCalo_EcalRecProducers_plugins_alpaka_EnergyComputationKernels_h
#define RecoLocalCalo_EcalRecProducers_plugins_alpaka_EnergyComputationKernels_h

#include <alpaka/alpaka.hpp>

#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

#include "KernelHelpers.h"

//#define DEBUG
//#define ECAL_RECO_ALPAKA_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit {

  using namespace cms::alpakatools;

  ALPAKA_STATIC_ACC_MEM_CONSTANT constexpr float ip10[] = {
      1.e5f, 1.e4f, 1.e3f, 1.e2f, 1.e1f, 1.e0f, 1.e-1f, 1.e-2f, 1.e-3f, 1.e-4};

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool checkUncalibRecHitFlag(uint32_t const& flags,
                                                             EcalUncalibratedRecHit::Flags flag) {
    return flags & (0x1 << flag);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void setFlag(uint32_t& flags, EcalRecHit::Flags flag) { flags |= (0x1 << flag); }

  class KernelCreateEcalRechit {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  EcalUncalibratedRecHitDeviceCollection::ConstView ebUncalibRecHits,
                                  EcalUncalibratedRecHitDeviceCollection::ConstView eeUncalibRecHits,
                                  EcalRecHitDeviceCollection::View ebRecHits,
                                  EcalRecHitDeviceCollection::View eeRecHits,
                                  EcalRecHitConditionsDevice::ConstView conditionsDev,
                                  EcalRecHitParametersDevice::ConstView parametersDev,
                                  // time, used for time dependent corrections
                                  edm::TimeValue_t const& eventTime,
                                  // configuration
                                  bool const killDeadChannels,
                                  bool const recoverEBIsolatedChannels,
                                  bool const recoverEEIsolatedChannels,
                                  bool const recoverEBVFE,
                                  bool const recoverEEVFE,
                                  bool const recoverEBFE,
                                  bool const recoverEEFE,
                                  float const EBLaserMIN,
                                  float const EELaserMIN,
                                  float const EBLaserMAX,
                                  float const EELaserMAX,
                                  uint32_t flagmask) const {
      auto const nchannelsEB = ebUncalibRecHits.size();
      auto const nchannelsEE = eeUncalibRecHits.size();
      auto const nchannels = nchannelsEB + nchannelsEE;

      for (auto ch : cms::alpakatools::uniform_elements(acc, nchannels)) {
        // set the output collection size scalars
        if (ch == 0) {
          ebRecHits.size() = nchannelsEB;
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
        //auto* chi2 = isEndcap ? eeRecHits.chi2() : ebRecHits.chi2();
        auto* flagBits = isEndcap ? eeRecHits.flagBits() : ebRecHits.flagBits();
        auto* extra = isEndcap ? eeRecHits.extra() : ebRecHits.extra();

        // only two values for ADC to GeV, EB or EE
        auto const adc2gev_to_use = isEndcap ? conditionsDev.adcToGeVConstantEE() : conditionsDev.adcToGeVConstantEB();

        // simple copy of input det id to output
        did[inputCh] = didCh[inputCh];

        auto const did_to_use = DetId{didCh[inputCh]};

        auto const isBarrel = did_to_use.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did_to_use.rawId())
                                       : nchannelsEB + ecal::reconstruction::hashedIndexEE(did_to_use.rawId());

        auto const intercalib = conditionsDev.intercalibConstants()[hashedId];

        auto const timeCalib = conditionsDev.timeCalibConstants()[hashedId];
        auto const timeOffset = isEndcap ? conditionsDev.timeOffsetConstantEE() : conditionsDev.timeOffsetConstantEB();

        //
        // AM: ideas
        //
        //    One possibility is to create the map of laser corrections once on CPU
        //    for all crystals and push them on GPU.
        //    Then only if the LS is different, update the laser correction
        //    The variation within a LS is not worth pursuing (<< 0.1% !!)
        //    and below the precision we can claim on the laser corrections (right?).
        //    This will save quite some time (also for the CPU version?)
        //

        int iLM = 1;

        if (isBarrel) {
          iLM = ecal::reconstruction::laserMonitoringRegionEB(did_to_use.rawId());
        } else {
          iLM = ecal::reconstruction::laserMonitoringRegionEE(did_to_use.rawId());
        }

        long long t_i = 0, t_f = 0;
        float p_i = 0, p_f = 0;
        auto const t1 = conditionsDev.laserAPDPNRatios_t1()[iLM - 1];
        auto const t2 = conditionsDev.laserAPDPNRatios_t2()[iLM - 1];
        auto const t3 = conditionsDev.laserAPDPNRatios_t3()[iLM - 1];
        auto const p1 = conditionsDev.laserAPDPNRatios_p1()[hashedId];
        auto const p2 = conditionsDev.laserAPDPNRatios_p2()[hashedId];
        auto const p3 = conditionsDev.laserAPDPNRatios_p3()[hashedId];

        // laser
        if (eventTime >= t1 && eventTime < t2) {
          t_i = t1;
          t_f = t2;
          p_i = p1;
          p_f = p2;
        } else if (eventTime >= t2 && eventTime <= t3) {
          t_i = t2;
          t_f = t3;
          p_i = p2;
          p_f = p3;
        } else if (eventTime < t1) {
          t_i = t1;
          t_f = t2;
          p_i = p1;
          p_f = p2;
        } else if (eventTime > t3) {
          t_i = t2;
          t_f = t3;
          p_i = p2;
          p_f = p3;
        }

        long long lt_i = 0, lt_f = 0;
        float lp_i = 0, lp_f = 0;
        auto const lt1 = conditionsDev.linearCorrections_t1()[iLM - 1];
        auto const lt2 = conditionsDev.linearCorrections_t2()[iLM - 1];
        auto const lt3 = conditionsDev.linearCorrections_t3()[iLM - 1];
        auto const lp1 = conditionsDev.linearCorrections_p1()[hashedId];
        auto const lp2 = conditionsDev.linearCorrections_p2()[hashedId];
        auto const lp3 = conditionsDev.linearCorrections_p3()[hashedId];

        // linear corrections
        if (eventTime >= lt1 && eventTime < lt2) {
          lt_i = lt1;
          lt_f = lt2;
          lp_i = lp1;
          lp_f = lp2;
        } else if (eventTime >= lt2 && eventTime <= lt3) {
          lt_i = lt2;
          lt_f = lt3;
          lp_i = lp2;
          lp_f = lp3;
        } else if (eventTime < lt1) {
          lt_i = lt1;
          lt_f = lt2;
          lp_i = lp1;
          lp_f = lp2;
        } else if (eventTime > lt3) {
          lt_i = lt2;
          lt_f = lt3;
          lp_i = lp2;
          lp_f = lp3;
        }

        // apdpnref and alpha
        auto const apdpnref = conditionsDev.laserAPDPNref()[hashedId];
        auto const alpha = conditionsDev.laserAlpha()[hashedId];

        // now calculate transparency correction
        float lasercalib = 1.;
        if (apdpnref != 0 && (t_i - t_f) != 0 && (lt_i - lt_f) != 0) {
          long long tt = eventTime;  // never subtract two unsigned!
          auto const interpolatedLaserResponse =
              p_i / apdpnref + static_cast<float>(tt - t_i) * (p_f - p_i) / (apdpnref * static_cast<float>(t_f - t_i));

          auto interpolatedLinearResponse =
              lp_i / apdpnref + static_cast<float>(tt - lt_i) * (lp_f - lp_i) /
                                    (apdpnref * static_cast<float>(lt_f - lt_i));  // FIXED BY FC

          if (interpolatedLinearResponse > 2.f || interpolatedLinearResponse < 0.1f) {
            interpolatedLinearResponse = 1.f;
          }
          if (interpolatedLaserResponse <= 0.) {
            lasercalib = 1.;
          } else {
            auto const interpolatedTransparencyResponse = interpolatedLaserResponse / interpolatedLinearResponse;
            lasercalib = 1.f / (std::pow(interpolatedTransparencyResponse, alpha) * interpolatedLinearResponse);
          }
        }

        //
        // Check for channels to be excluded from reconstruction
        //
        // Default energy not to be updated if "channelStatusToBeExcluded"
        // Exploited later by the module "EcalRecHitConvertGPU2CPUFormat"
        energy[inputCh] = -1;  //un-physical default

        // default values for the flags
        flagBits[inputCh] = 0;
        extra[inputCh] = 0;

        auto const dbChStatus = static_cast<EcalChannelStatusCode::Code>(conditionsDev.channelStatus()[hashedId] &
                                                                         EcalChannelStatusCode::chStatusMask);
        auto const& exclChStatCodes = parametersDev.channelStatusCodesToBeExcluded();
        if (exclChStatCodes[dbChStatus]) {
          // skip the channel if the channel status bit is set and should be excluded
          continue;
        }

        // Take our association map of dbChStatuses-> recHit flagbits and return the appropriate flagbit word
        auto const& recoFlagBits = parametersDev.recoFlagBits()[dbChStatus];
        flagBits[inputCh] = recoFlagBits;

        if ((flagmask & recoFlagBits) && killDeadChannels) {
          // skip this channel
          continue;
        }

        //
        // multiply the adc counts with factors to get the energy in GeV
        //
        energy[inputCh] = amplitude[inputCh] * adc2gev_to_use * intercalib * lasercalib;

        // time
        time[inputCh] = jitter[inputCh] * ecalPh1::Samp_Period + timeCalib + timeOffset;

        // NB: calculate the "flagBits extra"  --> not really "flags", but actually an encoded version of energy uncertainty, time unc., ...

        //
        // extra packing ...
        //
        uint32_t extravar = 0;

        // chi2
        // truncate
        auto const chi2 = chi2_in[inputCh] > 64 ? 64 : chi2_in[inputCh];
        // use 7 bits
        uint32_t const rawChi2 = lround(chi2 / 64. * ((1 << 7) - 1));
        extravar = ecal::reconstruction::rechitSetMasked(extravar, rawChi2, 0, 7);

        // energy uncertainty
        auto const energyError = amplitudeError[inputCh] * adc2gev_to_use * intercalib * lasercalib;
        uint32_t rawEnergy = 0;
        if (energyError > 0.001) {
          uint16_t const exponent = static_cast<uint16_t>(ecal::reconstruction::rechitGetPower10(energyError));
          uint16_t significand = lround(energyError * ip10[exponent]);
          // use 13 bits (3 exponent, 10 significand)
          rawEnergy = exponent << 10 | significand;
        }
        extravar = ecal::reconstruction::rechitSetMasked(extravar, rawEnergy, 8, 13);

        // time uncertainty directly from uncalib rechit
        uint8_t const timeErrBits = aux[inputCh] & 0xFF;
        extravar = ecal::reconstruction::rechitSetMasked(extravar, timeErrBits, 24, 8);

        //
        // set output extra variable
        //
        extra[inputCh] = extravar;

        //
        // additional flags setting
        // using correctly the flags as calculated at the UncalibRecHit stage
        //
        bool good = true;
        if (checkUncalibRecHitFlag(flags_in[inputCh], EcalUncalibratedRecHit::Flags::kLeadingEdgeRecovered)) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kLeadingEdgeRecovered);
          good = false;
        }
        if (checkUncalibRecHitFlag(flags_in[inputCh], EcalUncalibratedRecHit::Flags::kSaturated)) {
          // leading edge recovery failed - still keep the information
          // about the saturation and do not flag as dead
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kSaturated);
          good = false;
        }
        if (checkUncalibRecHitFlag(flags_in[inputCh], EcalUncalibratedRecHit::Flags::kOutOfTime)) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kOutOfTime);
          good = false;
        }
        if (checkUncalibRecHitFlag(flags_in[inputCh], EcalUncalibratedRecHit::Flags::kPoorReco)) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kPoorReco);
          good = false;
        }
        if (checkUncalibRecHitFlag(flags_in[inputCh], EcalUncalibratedRecHit::Flags::kHasSwitchToGain6)) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kHasSwitchToGain6);
        }
        if (checkUncalibRecHitFlag(flags_in[inputCh], EcalUncalibratedRecHit::Flags::kHasSwitchToGain1)) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kHasSwitchToGain1);
        }

        if (good) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kGood);
        }

        if ((isBarrel && (lasercalib < EBLaserMIN || lasercalib > EBLaserMAX)) ||
            (!isBarrel && (lasercalib < EELaserMIN || lasercalib > EELaserMAX))) {
          setFlag(flagBits[inputCh], EcalRecHit::Flags::kPoorCalib);
        }

        // recover, killing, and other stuff
        //
        // Structure:
        //  EB
        //  EE
        //
        //  - single MVA
        //  - democratic sharing
        //  - kill all the other cases

        // recoverable channel status codes
        if (dbChStatus == EcalChannelStatusCode::Code::kFixedG0 ||
            dbChStatus == EcalChannelStatusCode::Code::kNonRespondingIsolated ||
            dbChStatus == EcalChannelStatusCode::Code::kDeadVFE) {
          bool is_Single = false;
          bool is_FE = false;
          bool is_VFE = false;

          if (dbChStatus == EcalChannelStatusCode::Code::kDeadVFE) {
            is_VFE = true;
          } else if (dbChStatus == EcalChannelStatusCode::Code::kDeadFE) {
            is_FE = true;
          } else {
            is_Single = true;
          }

          // EB
          if (isBarrel) {
            if (is_Single || is_FE || is_VFE) {
              if (is_Single && (recoverEBIsolatedChannels || !killDeadChannels)) {
                // single MVA
                // TODO
              } else if (is_FE && (recoverEBFE || !killDeadChannels)) {
                // democratic sharing
                // TODO
              } else {
                // kill all the other cases
                energy[inputCh] = 0.;
                // TODO: set flags
              }
            }
          } else {  // EE
            if (is_Single || is_FE || is_VFE) {
              if (is_Single && (recoverEEIsolatedChannels || !killDeadChannels)) {
                // single MVA
                // TODO
              } else if (is_FE && (recoverEEFE || !killDeadChannels)) {
                // democratic sharing
                // TODO
              } else {
                // kill all the other cases
                energy[inputCh] = 0.;
                // TODO: set flags
              }
            }
          }
        }
      }  // end channel
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit

#endif  // RecoLocalCalo_EcalRecProducers_plugins_alpaka_EnergyComputationKernels_h
