#include <cuda.h>

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"

#include "EcalRecHitBuilderKernels.h"
#include "KernelHelpers.h"

namespace ecal {
  namespace rechit {

    // uncalibrecHit flags
    enum UncalibRecHitFlags {
      kGood = -1,  // channel is good (mutually exclusive with other states)  setFlagBit(kGood) reset flags_ to zero
      kPoorReco,   // channel has been badly reconstructed (e.g. bad shape, bad chi2 etc.)
      kSaturated,  // saturated channel
      kOutOfTime,  // channel out of time
      kLeadingEdgeRecovered,  // saturated channel: energy estimated from the leading edge before saturation
      kHasSwitchToGain6,      // at least one data frame is in G6
      kHasSwitchToGain1       // at least one data frame is in G1
    };

    // recHit flags
    enum RecHitFlags {
      RecHitFlags_kGood = 0,  // channel ok, the energy and time measurement are reliable
      RecHitFlags_kPoorReco,  // the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
      RecHitFlags_kOutOfTime,  // the energy is available from the UncalibRecHit (sync reco), but the event is out of time
      RecHitFlags_kFaultyHardware,  // The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
      RecHitFlags_kNoisy,      // the channel is very noisy
      RecHitFlags_kPoorCalib,  // the energy is available from the UncalibRecHit, but the calibration of the channel is poor
      RecHitFlags_kSaturated,             // saturated channel (recovery not tried)
      RecHitFlags_kLeadingEdgeRecovered,  // saturated channel: energy estimated from the leading edge before saturation
      RecHitFlags_kNeighboursRecovered,   // saturated/isolated dead: energy estimated from neighbours
      RecHitFlags_kTowerRecovered,        // channel in TT with no data link, info retrieved from Trigger Primitive
      RecHitFlags_kDead,                  // channel is dead and any recovery fails
      RecHitFlags_kKilled,                // MC only flag: the channel is killed in the real detector
      RecHitFlags_kTPSaturated,           // the channel is in a region with saturated TP
      RecHitFlags_kL1SpikeFlag,           // the channel is in a region with TP with sFGVB = 0
      RecHitFlags_kWeird,                 // the signal is believed to originate from an anomalous deposit (spike)
      RecHitFlags_kDiWeird,               // the signal is anomalous, and neighbors another anomalous signal
      RecHitFlags_kHasSwitchToGain6,      // at least one data frame is in G6
      RecHitFlags_kHasSwitchToGain1,      // at least one data frame is in G1
      //
      RecHitFlags_kUnknown  // to ease the interface with functions returning flags.
    };

    // status code
    enum EcalChannelStatusCode_Code {
      kOk = 0,
      kDAC,
      kNoLaser,
      kNoisy,
      kNNoisy,
      kNNNoisy,
      kNNNNoisy,
      kNNNNNoisy,
      kFixedG6,
      kFixedG1,
      kFixedG0,
      kNonRespondingIsolated,
      kDeadVFE,
      kDeadFE,
      kNoDataNoTP
    };

    __global__ void kernel_create_ecal_rehit(
        // configuration
        int const* ChannelStatusToBeExcluded,
        uint32_t ChannelStatusToBeExcludedSize,
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
        // for flags setting
        int const* expanded_v_DB_reco_flags,  // FIXME AM: to be checked
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
        uint32_t* didEB,
        uint32_t* didEE,
        ::ecal::reco::StorageScalarType* energyEB,  // in energy [GeV]
        ::ecal::reco::StorageScalarType* energyEE,  // in energy [GeV]
        ::ecal::reco::StorageScalarType* timeEB,
        ::ecal::reco::StorageScalarType* timeEE,
        ::ecal::reco::StorageScalarType* chi2EB,
        ::ecal::reco::StorageScalarType* chi2EE,
        uint32_t* flagBitsEB,
        uint32_t* flagBitsEE,
        uint32_t* extraEB,
        uint32_t* extraEE,
        // other
        int const nchannels,
        uint32_t const nChannelsBarrel,
        uint32_t const offsetForHashes) {
      //
      //    NB: energy   "type_wrapper<reco::StorageScalarType, L>::type" most likely std::vector<float>
      //

      for (int ch = threadIdx.x + blockDim.x * blockIdx.x; ch < nchannels; ch += blockDim.x * gridDim.x) {
        bool isEndcap = (ch >= nChannelsBarrel);

        int const inputCh = isEndcap ? ch - nChannelsBarrel : ch;

        uint32_t const* didCh = isEndcap ? did_ee : did_eb;

        // arrange to access the right ptrs
#define ARRANGE(var) auto* var = isEndcap ? var##EE : var##EB
        ARRANGE(did);
        ARRANGE(energy);
        ARRANGE(chi2);
        ARRANGE(flagBits);
        ARRANGE(extra);
#undef ARRANGE

        // only two values, EB or EE
        // AM : FIXME : why not using "isBarrel" ?    isBarrel ? adc2gev[0] : adc2gev[1]
        float adc2gev_to_use = isEndcap ? adc2gev[1]   // ee
                                        : adc2gev[0];  // eb

        // first EB and then EE

        ::ecal::reco::StorageScalarType const* amplitude = isEndcap ? amplitude_ee : amplitude_eb;

        ::ecal::reco::StorageScalarType const* chi2_in = isEndcap ? chi2_ee : chi2_eb;

        uint32_t const* flags_in = isEndcap ? flags_ee : flags_eb;

        // simple copy
        did[inputCh] = didCh[inputCh];

        auto const did_to_use = DetId{didCh[inputCh]};

        auto const isBarrel = did_to_use.subdetId() == EcalBarrel;
        auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did_to_use.rawId())
                                       : offsetForHashes + ecal::reconstruction::hashedIndexEE(did_to_use.rawId());

        float const intercalib_to_use = intercalib[hashedId];

        // get laser coefficient
        float lasercalib = 1.;

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
          iLM = ecal::reconstruction::laser_monitoring_region_EB(did_to_use.rawId());
        } else {
          iLM = ecal::reconstruction::laser_monitoring_region_EE(did_to_use.rawId());
        }

        long long t_i = 0, t_f = 0;
        float p_i = 0, p_f = 0;
        long long lt_i = 0, lt_f = 0;
        float lp_i = 0, lp_f = 0;

        // laser
        if (event_time >= t1[iLM - 1] && event_time < t2[iLM - 1]) {
          t_i = t1[iLM - 1];
          t_f = t2[iLM - 1];
          p_i = p1[hashedId];
          p_f = p2[hashedId];
        } else if (event_time >= t2[iLM - 1] && event_time <= t3[iLM - 1]) {
          t_i = t2[iLM - 1];
          t_f = t3[iLM - 1];
          p_i = p2[hashedId];
          p_f = p3[hashedId];
        } else if (event_time < t1[iLM - 1]) {
          t_i = t1[iLM - 1];
          t_f = t2[iLM - 1];
          p_i = p1[hashedId];
          p_f = p2[hashedId];

        } else if (event_time > t3[iLM - 1]) {
          t_i = t2[iLM - 1];
          t_f = t3[iLM - 1];
          p_i = p2[hashedId];
          p_f = p3[hashedId];
        }

        // linear corrections
        if (event_time >= lt1[iLM - 1] && event_time < lt2[iLM - 1]) {
          lt_i = lt1[iLM - 1];
          lt_f = lt2[iLM - 1];
          lp_i = lp1[hashedId];
          lp_f = lp2[hashedId];
        } else if (event_time >= lt2[iLM - 1] && event_time <= lt3[iLM - 1]) {
          lt_i = lt2[iLM - 1];
          lt_f = lt3[iLM - 1];
          lp_i = lp2[hashedId];
          lp_f = lp3[hashedId];
        } else if (event_time < lt1[iLM - 1]) {
          lt_i = lt1[iLM - 1];
          lt_f = lt2[iLM - 1];
          lp_i = lp1[hashedId];
          lp_f = lp2[hashedId];

        } else if (event_time > lt3[iLM - 1]) {
          lt_i = lt2[iLM - 1];
          lt_f = lt3[iLM - 1];
          lp_i = lp2[hashedId];
          lp_f = lp3[hashedId];
        }

        // apdpnref and alpha
        float apdpnref = apdpnrefs[hashedId];
        float alpha = alphas[hashedId];

        // now calculate transparency correction
        if (apdpnref != 0 && (t_i - t_f) != 0 && (lt_i - lt_f) != 0) {
          long long tt = event_time;  // never subtract two unsigned!
          float interpolatedLaserResponse =
              p_i / apdpnref + float(tt - t_i) * (p_f - p_i) / (apdpnref * float(t_f - t_i));

          float interpolatedLinearResponse =
              lp_i / apdpnref + float(tt - lt_i) * (lp_f - lp_i) / (apdpnref * float(lt_f - lt_i));  // FIXED BY FC

          if (interpolatedLinearResponse > 2.f || interpolatedLinearResponse < 0.1f) {
            interpolatedLinearResponse = 1.f;
          }
          if (interpolatedLaserResponse <= 0.) {
            // AM :  how the heck is it possible?
            //             interpolatedLaserResponse = 0.0001;
            lasercalib = 1.;

          } else {
            float interpolatedTransparencyResponse = interpolatedLaserResponse / interpolatedLinearResponse;

            // ... and now this:
            lasercalib = 1.f / (std::pow(interpolatedTransparencyResponse, alpha) * interpolatedLinearResponse);
          }
        }

        //
        // Check for channels to be excluded from reconstruction
        //
        //
        // Default energy? Not to be updated if "ChannelStatusToBeExcluded"
        // Exploited later by the module "EcalRecHitConvertGPU2CPUFormat"
        //
        energy[inputCh] = -1;  //---- AM: default, un-physical, ok

        // truncate the chi2
        if (chi2_in[inputCh] > 64)
          chi2[inputCh] = 64;
        else
          chi2[inputCh] = chi2_in[inputCh];

        // default values for the flags
        flagBits[inputCh] = 0;
        extra[inputCh] = 0;

        static const int chStatusMask = 0x1f;
        // ChannelStatusToBeExcluded is a "int" then I put "dbstatus" to be the same
        int dbstatus = EcalChannelStatusCode_Code((status[hashedId]) & chStatusMask);
        if (ChannelStatusToBeExcludedSize != 0) {
          bool skip_this_channel = false;
          for (int ich_to_check = 0; ich_to_check < ChannelStatusToBeExcludedSize; ich_to_check++) {
            if (ChannelStatusToBeExcluded[ich_to_check] == dbstatus) {
              skip_this_channel = true;
              break;
            }
          }
          if (skip_this_channel) {
            // skip this channel
            continue;
          }
        }

        // Take our association map of dbstatuses-> recHit flagbits and return the apporpriate flagbit word

        //
        // AM: get the smaller "flagbit_counter" with match
        //

        uint32_t temporary_flagBits = 0;

        int iterator_flags = 0;
        bool need_to_exit = false;
        int flagbit_counter = 0;
        while (!need_to_exit) {
          iterator_flags = 0;
          for (unsigned int i = 0; i != expanded_v_DB_reco_flagsSize; ++i) {
            // check the correct "flagbit"
            if (expanded_flagbit_v_DB_reco_flags[i] == flagbit_counter) {
              for (unsigned int j = 0; j < expanded_Sizes_v_DB_reco_flags[i]; j++) {
                if (expanded_v_DB_reco_flags[iterator_flags] == dbstatus) {
                  temporary_flagBits = 0x1 << expanded_flagbit_v_DB_reco_flags[i];
                  need_to_exit = true;
                  break;  // also from the big loop!!!
                }
                iterator_flags++;
              }
            } else {
              // if not, got to the next bunch directly
              iterator_flags += expanded_Sizes_v_DB_reco_flags[i];
            }

            if (need_to_exit) {
              break;
            }
          }
          flagbit_counter += 1;
        }

        flagBits[inputCh] = temporary_flagBits;

        if ((flagmask & temporary_flagBits) && killDeadChannels) {
          // skip this channel
          continue;
        }

        //
        // multiply the adc counts with factors to get GeV
        //

        //         energy[ch] = amplitude[inputCh] * adc2gev_to_use * intercalib_to_use ;
        energy[inputCh] = amplitude[inputCh] * adc2gev_to_use * intercalib_to_use * lasercalib;

        // Time is not saved so far, FIXME
        //         time[ch] = time_in[inputCh];

        // NB: calculate the "flagBits extra"  --> not really "flags", but actually an encoded version of energy uncertainty, time unc., ...

        //
        // extra packing ...
        //

        uint32_t offset;
        uint32_t width;
        uint32_t value;

        float chi2_temp = chi2[inputCh];
        if (chi2_temp > 64)
          chi2_temp = 64;
        // use 7 bits
        uint32_t rawChi2 = lround(chi2_temp / 64. * ((1 << 7) - 1));

        offset = 0;
        width = 7;
        value = 0;

        uint32_t mask = ((1 << width) - 1) << offset;
        value &= ~mask;
        value |= (rawChi2 & ((1U << width) - 1)) << offset;

        // rawEnergy is actually "error" !!!
        uint32_t rawEnergy = 0;

        // AM: FIXME: this is not propagated currently to the uncalibrecHit collection SOA
        //            if you want to store this in "extra", we need first to add it to the uncalibrecHit results
        //            then it will be something like the following
        //         amplitudeError[inputCh] * adc2gev_to_use * intercalib_to_use * lasercalib
        //
        //

        float amplitudeError_ch = 0.;  // amplitudeError[ch];

        if (amplitudeError_ch > 0.001) {
          static constexpr float p10[] = {1.e-2f, 1.e-1f, 1.f, 1.e1f, 1.e2f, 1.e3f, 1.e4f, 1.e5f, 1.e6f};
          int b = amplitudeError_ch < p10[4] ? 0 : 5;
          for (; b < 9; ++b)
            if (amplitudeError_ch < p10[b])
              break;

          uint16_t exponent = b;

          static constexpr float ip10[] = {1.e5f, 1.e4f, 1.e3f, 1.e2f, 1.e1f, 1.e0f, 1.e-1f, 1.e-2f, 1.e-3f, 1.e-4};
          uint16_t significand = lround(amplitudeError_ch * ip10[exponent]);
          // use 13 bits (3 exponent, 10 significand)
          rawEnergy = exponent << 10 | significand;
        }

        offset = 8;
        width = 13;
        // value from last change, ok

        mask = ((1 << width) - 1) << offset;
        value &= ~mask;
        value |= (rawEnergy & ((1U << width) - 1)) << offset;

        uint32_t jitterErrorBits = 0;
        jitterErrorBits = jitterErrorBits & 0xFF;

        offset = 24;
        width = 8;
        // value from last change, ok

        mask = ((1 << width) - 1) << offset;
        value &= ~mask;
        value |= (jitterErrorBits & ((1U << width) - 1)) << offset;

        //
        // now finally set "extra[ch]"
        //
        extra[inputCh] = value;

        //
        // additional flags setting
        //
        // using correctly the flags as calculated at the UncalibRecHit stage
        //
        // Now fill flags

        bool good = true;

        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kLeadingEdgeRecovered))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kLeadingEdgeRecovered));
          good = false;
        }

        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kSaturated))) {
          // leading edge recovery failed - still keep the information
          // about the saturation and do not flag as dead
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kSaturated));
          good = false;
        }

        //
        // AM: why do we have two tests one after the other checking almost the same thing???
        // Please clean up the code, ... also the original one!
        //
        // uncalibRH.isSaturated() --->
        //
        //                                   bool EcalUncalibratedRecHit::isSaturated() const {
        //                                     return EcalUncalibratedRecHit::checkFlag(kSaturated);
        //                                   }
        //
        //

        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kSaturated))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kSaturated));
          good = false;
        }

        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kOutOfTime))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kOutOfTime));
          good = false;
        }
        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kPoorReco))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kPoorReco));
          good = false;
        }
        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kHasSwitchToGain6))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kHasSwitchToGain6));
        }
        if (flags_in[inputCh] & (0x1 << (UncalibRecHitFlags::kHasSwitchToGain1))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kHasSwitchToGain1));
        }

        if (good) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kGood));
        }

        if ((isBarrel && (lasercalib < EBLaserMIN || lasercalib > EBLaserMAX)) || (!isBarrel && (lasercalib < EELaserMIN || lasercalib > EELaserMAX))) {
          flagBits[inputCh] |= (0x1 << (RecHitFlags::RecHitFlags_kPoorCalib));
        }

        // recover, killing, and other stuff

        //
        // Structure:
        //  EB
        //  EE
        //
        //
        //  - single MVA
        //  - democratic sharing
        //  - kill all the other cases
        //

        bool is_Single = false;
        bool is_FE = false;
        bool is_VFE = false;

        bool is_recoverable = false;  // DetIdToBeRecovered

        if (dbstatus == 10 || dbstatus == 11 || dbstatus == 12) {
          is_recoverable = true;
        }

        if (is_recoverable) {
          if (dbstatus == EcalChannelStatusCode_Code::kDeadVFE) {
            is_VFE = true;
          } else if (dbstatus == EcalChannelStatusCode_Code::kDeadVFE) {
            is_FE = true;
          } else {
            is_Single = true;
          }

          // EB
          if (isBarrel) {
            if (is_Single || is_FE || is_VFE) {
              // single MVA
              if (is_Single && (recoverEBIsolatedChannels || !killDeadChannels)) {
              }
              // decmocratic sharing
              else if (is_FE && (recoverEBFE || !killDeadChannels)) {
              }
              // kill all the other cases
              else {
                energy[inputCh] = 0.;  // Need to set also the flags ...
              }
            }
          }
          // EE
          else {
            if (is_Single || is_FE || is_VFE) {
              // single MVA
              if (is_Single && (recoverEBIsolatedChannels || !killDeadChannels)) {
              }
              // decmocratic sharing
              else if (is_FE && (recoverEBFE || !killDeadChannels)) {
                //
                //  Code is definitely too long ...
                //

              }
              // kill all the other cases
              else {
                energy[inputCh] = 0.;  // Need to set also the flags ...
              }
            }
          }
        }

      }  // end channel
    }

    // host version, to be called by the plugin
    void create_ecal_rehit(EventInputDataGPU const& eventInputGPU,
                           EventOutputDataGPU& eventOutputGPU,
                           //     eventDataForScratchGPU_,
                           ConditionsProducts const& conditions,
                           ConfigurationParameters const& configParameters,
                           uint32_t const nChannelsBarrel,
                           edm::TimeValue_t const event_time,
                           cudaStream_t cudaStream) {
      int nchannels = eventInputGPU.ebUncalibRecHits.size + eventInputGPU.eeUncalibRecHits.size;

      unsigned int nchannels_per_block = 16;
      unsigned int threads_min = nchannels_per_block;
      unsigned int blocks_min = (nchannels + threads_min - 1) / threads_min;  // TEST : to be optimized (AM)

      //
      // kernel create rechit
      //

      kernel_create_ecal_rehit<<<blocks_min, threads_min, 0, cudaStream>>>(
          // configuration
          configParameters.ChannelStatusToBeExcluded,
          configParameters.ChannelStatusToBeExcludedSize,
          configParameters.killDeadChannels,
          configParameters.recoverEBIsolatedChannels,
          configParameters.recoverEEIsolatedChannels,
          configParameters.recoverEBVFE,
          configParameters.recoverEEVFE,
          configParameters.recoverEBFE,
          configParameters.recoverEEFE,
          configParameters.EBLaserMIN,
          configParameters.EELaserMIN,
          configParameters.EBLaserMAX,
          configParameters.EELaserMAX,
          // for flags setting
          configParameters.expanded_v_DB_reco_flags,
          configParameters.expanded_Sizes_v_DB_reco_flags,
          configParameters.expanded_flagbit_v_DB_reco_flags,
          configParameters.expanded_v_DB_reco_flagsSize,
          configParameters.flagmask,
          // conditions
          conditions.ADCToGeV.adc2gev,
          conditions.Intercalib.values,
          conditions.ChannelStatus.status,
          conditions.LaserAPDPNRatiosRef.values,
          conditions.LaserAlphas.values,
          // input for transparency corrections
          conditions.LaserAPDPNRatios.p1,
          conditions.LaserAPDPNRatios.p2,
          conditions.LaserAPDPNRatios.p3,
          conditions.LaserAPDPNRatios.t1,
          conditions.LaserAPDPNRatios.t2,
          conditions.LaserAPDPNRatios.t3,
          // input for linear corrections
          conditions.LinearCorrections.p1,
          conditions.LinearCorrections.p2,
          conditions.LinearCorrections.p3,
          conditions.LinearCorrections.t1,
          conditions.LinearCorrections.t2,
          conditions.LinearCorrections.t3,
          // time, used for time dependent corrections
          event_time,
          // input
          eventInputGPU.ebUncalibRecHits.did.get(),
          eventInputGPU.eeUncalibRecHits.did.get(),
          eventInputGPU.ebUncalibRecHits.amplitude.get(),
          eventInputGPU.eeUncalibRecHits.amplitude.get(),
          eventInputGPU.ebUncalibRecHits.jitter.get(),
          eventInputGPU.eeUncalibRecHits.jitter.get(),
          eventInputGPU.ebUncalibRecHits.chi2.get(),
          eventInputGPU.eeUncalibRecHits.chi2.get(),
          eventInputGPU.ebUncalibRecHits.flags.get(),
          eventInputGPU.eeUncalibRecHits.flags.get(),
          // output
          eventOutputGPU.recHitsEB.did.get(),
          eventOutputGPU.recHitsEE.did.get(),
          eventOutputGPU.recHitsEB.energy.get(),
          eventOutputGPU.recHitsEE.energy.get(),
          eventOutputGPU.recHitsEB.time.get(),
          eventOutputGPU.recHitsEE.time.get(),
          eventOutputGPU.recHitsEB.chi2.get(),
          eventOutputGPU.recHitsEE.chi2.get(),
          eventOutputGPU.recHitsEB.flagBits.get(),
          eventOutputGPU.recHitsEE.flagBits.get(),
          eventOutputGPU.recHitsEB.extra.get(),
          eventOutputGPU.recHitsEE.extra.get(),
          // other
          nchannels,
          nChannelsBarrel,
          conditions.offsetForHashes);
    }

  }  // namespace rechit

}  // namespace ecal
