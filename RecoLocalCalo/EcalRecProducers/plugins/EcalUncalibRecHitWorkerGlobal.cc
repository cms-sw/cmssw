#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerGlobal.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

EcalUncalibRecHitWorkerGlobal::EcalUncalibRecHitWorkerGlobal(const edm::ParameterSet& ps, edm::ConsumesCollector& c)
    : EcalUncalibRecHitWorkerRunOneDigiBase(ps, c),
      tokenPeds_(c.esConsumes<EcalPedestals, EcalPedestalsRcd>()),
      tokenGains_(c.esConsumes<EcalGainRatios, EcalGainRatiosRcd>()),
      tokenGrps_(c.esConsumes<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd>()),
      tokenWgts_(c.esConsumes<EcalTBWeights, EcalTBWeightsRcd>()),
      testbeamEEShape(EEShape(true)),
      testbeamEBShape(EBShape(true)),
      tokenSampleMask_(c.esConsumes<EcalSampleMask, EcalSampleMaskRcd>()),
      tokenTimeCorrBias_(c.esConsumes<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd>()),
      tokenItime_(c.esConsumes<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd>()),
      tokenOfftime_(c.esConsumes<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd>()) {
  // ratio method parameters
  EBtimeFitParameters_ = ps.getParameter<std::vector<double>>("EBtimeFitParameters");
  EEtimeFitParameters_ = ps.getParameter<std::vector<double>>("EEtimeFitParameters");
  EBamplitudeFitParameters_ = ps.getParameter<std::vector<double>>("EBamplitudeFitParameters");
  EEamplitudeFitParameters_ = ps.getParameter<std::vector<double>>("EEamplitudeFitParameters");
  EBtimeFitLimits_.first = ps.getParameter<double>("EBtimeFitLimits_Lower");
  EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
  EEtimeFitLimits_.first = ps.getParameter<double>("EEtimeFitLimits_Lower");
  EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
  EBtimeConstantTerm_ = ps.getParameter<double>("EBtimeConstantTerm");
  EBtimeNconst_ = ps.getParameter<double>("EBtimeNconst");
  EEtimeConstantTerm_ = ps.getParameter<double>("EEtimeConstantTerm");
  EEtimeNconst_ = ps.getParameter<double>("EEtimeNconst");
  outOfTimeThreshG12pEB_ = ps.getParameter<double>("outOfTimeThresholdGain12pEB");
  outOfTimeThreshG12mEB_ = ps.getParameter<double>("outOfTimeThresholdGain12mEB");
  outOfTimeThreshG61pEB_ = ps.getParameter<double>("outOfTimeThresholdGain61pEB");
  outOfTimeThreshG61mEB_ = ps.getParameter<double>("outOfTimeThresholdGain61mEB");
  outOfTimeThreshG12pEE_ = ps.getParameter<double>("outOfTimeThresholdGain12pEE");
  outOfTimeThreshG12mEE_ = ps.getParameter<double>("outOfTimeThresholdGain12mEE");
  outOfTimeThreshG61pEE_ = ps.getParameter<double>("outOfTimeThresholdGain61pEE");
  outOfTimeThreshG61mEE_ = ps.getParameter<double>("outOfTimeThresholdGain61mEE");
  amplitudeThreshEB_ = ps.getParameter<double>("amplitudeThresholdEB");
  amplitudeThreshEE_ = ps.getParameter<double>("amplitudeThresholdEE");

  // spike threshold
  ebSpikeThresh_ = ps.getParameter<double>("ebSpikeThreshold");

  ebPulseShape_ = ps.getParameter<std::vector<double>>("ebPulseShape");
  eePulseShape_ = ps.getParameter<std::vector<double>>("eePulseShape");

  // chi2 parameters
  kPoorRecoFlagEB_ = ps.getParameter<bool>("kPoorRecoFlagEB");
  kPoorRecoFlagEE_ = ps.getParameter<bool>("kPoorRecoFlagEE");

  chi2ThreshEB_ = ps.getParameter<double>("chi2ThreshEB_");
  chi2ThreshEE_ = ps.getParameter<double>("chi2ThreshEE_");
  EBchi2Parameters_ = ps.getParameter<std::vector<double>>("EBchi2Parameters");
  EEchi2Parameters_ = ps.getParameter<std::vector<double>>("EEchi2Parameters");
}

void EcalUncalibRecHitWorkerGlobal::set(const edm::EventSetup& es) {
  // common setup
  gains_ = es.getHandle(tokenGains_);
  peds_ = es.getHandle(tokenPeds_);

  // for the weights method
  grps_ = es.getHandle(tokenGrps_);
  wgts_ = es.getHandle(tokenWgts_);

  // which of the samples need be used
  sampleMaskHand_ = es.getHandle(tokenSampleMask_);

  // for the ratio method

  itime_ = es.getHandle(tokenItime_);
  offtime_ = es.getHandle(tokenOfftime_);

  // for the time correction methods
  timeCorrBias_ = es.getHandle(tokenTimeCorrBias_);

  // for the DB Ecal Pulse Sim Shape
  testbeamEEShape.setEventSetup(es);
  testbeamEBShape.setEventSetup(es);
}

// check saturation: 5 samples with gainId = 0
template <class C>
int EcalUncalibRecHitWorkerGlobal::isSaturated(const C& dataFrame) {
  //bool saturated_ = 0;
  int cnt;
  for (int j = 0; j < C::MAXSAMPLES - 5; ++j) {
    cnt = 0;
    for (int i = j; i < (j + 5) && i < C::MAXSAMPLES; ++i) {
      if (dataFrame.sample(i).gainId() == 0)
        ++cnt;
    }
    if (cnt == 5)
      return j - 1;  // the last unsaturated sample
  }
  return -1;  // no saturation found
}

/**
 * Amplitude-dependent time corrections; EE and EB have separate corrections:
 * EXtimeCorrAmplitudes (ADC) and EXtimeCorrShifts (ns) need to have the same number of elements
 * Bins must be ordered in amplitude. First-last bins take care of under-overflows.
 *
 * The algorithm is the same for EE and EB, only the correction vectors are different.
 *
 * @return Jitter (in clock cycles) which will be added to UncalibRechit.setJitter(), 0 if no correction is applied.
 */
double EcalUncalibRecHitWorkerGlobal::timeCorrection(float ampli,
                                                     const std::vector<float>& amplitudeBins,
                                                     const std::vector<float>& shiftBins) {
  // computed initially in ns. Than turned in the BX's, as
  // EcalUncalibratedRecHit need be.
  double theCorrection = 0;

  // sanity check for arrays
  if (amplitudeBins.empty()) {
    edm::LogError("EcalRecHitError") << "timeCorrAmplitudeBins is empty, forcing no time bias corrections.";

    return 0;
  }

  if (amplitudeBins.size() != shiftBins.size()) {
    edm::LogError("EcalRecHitError") << "Size of timeCorrAmplitudeBins different from "
                                        "timeCorrShiftBins. Forcing no time bias corrections. ";

    return 0;
  }

  int myBin = -1;
  for (int bin = 0; bin < (int)amplitudeBins.size(); bin++) {
    if (ampli > amplitudeBins.at(bin)) {
      myBin = bin;
    } else {
      break;
    }
  }

  if (myBin == -1) {
    theCorrection = shiftBins.at(0);
  } else if (myBin == ((int)(amplitudeBins.size() - 1))) {
    theCorrection = shiftBins.at(myBin);
  } else if (-1 < myBin && myBin < ((int)amplitudeBins.size() - 1)) {
    // interpolate linearly between two assingned points
    theCorrection = (shiftBins.at(myBin + 1) - shiftBins.at(myBin));
    theCorrection *=
        (((double)ampli) - amplitudeBins.at(myBin)) / (amplitudeBins.at(myBin + 1) - amplitudeBins.at(myBin));
    theCorrection += shiftBins.at(myBin);
  } else {
    edm::LogError("EcalRecHitError") << "Assigning time correction impossible. Setting it to 0 ";
    theCorrection = 0.;
  }

  // convert ns into clocks
  return theCorrection / 25.;
}

bool EcalUncalibRecHitWorkerGlobal::run(const edm::Event& evt,
                                        const EcalDigiCollection::const_iterator& itdg,
                                        EcalUncalibratedRecHitCollection& result) {
  DetId detid(itdg->id());

  const EcalSampleMask* sampleMask_ = sampleMaskHand_.product();

  // intelligence for recHit computation
  EcalUncalibratedRecHit uncalibRecHit;

  const EcalPedestals::Item* aped = nullptr;
  const EcalMGPAGainRatio* aGain = nullptr;
  const EcalXtalGroupId* gid = nullptr;
  float offsetTime = 0;

  if (detid.subdetId() == EcalEndcap) {
    unsigned int hashedIndex = EEDetId(detid).hashedIndex();
    aped = &peds_->endcap(hashedIndex);
    aGain = &gains_->endcap(hashedIndex);
    gid = &grps_->endcap(hashedIndex);
    offsetTime = offtime_->getEEValue();
  } else {
    unsigned int hashedIndex = EBDetId(detid).hashedIndex();
    aped = &peds_->barrel(hashedIndex);
    aGain = &gains_->barrel(hashedIndex);
    gid = &grps_->barrel(hashedIndex);
    offsetTime = offtime_->getEBValue();
  }

  pedVec[0] = aped->mean_x12;
  pedVec[1] = aped->mean_x6;
  pedVec[2] = aped->mean_x1;
  pedRMSVec[0] = aped->rms_x12;
  pedRMSVec[1] = aped->rms_x6;
  pedRMSVec[2] = aped->rms_x1;
  gainRatios[0] = 1.;
  gainRatios[1] = aGain->gain12Over6();
  gainRatios[2] = aGain->gain6Over1() * aGain->gain12Over6();

  // compute the right bin of the pulse shape using time calibration constants
  EcalTimeCalibConstantMap::const_iterator it = itime_->find(detid);
  EcalTimeCalibConstant itimeconst = 0;
  if (it != itime_->end()) {
    itimeconst = (*it);
  } else {
    edm::LogError("EcalRecHitError") << "No time intercalib const found for xtal " << detid.rawId()
                                     << "! something wrong with EcalTimeCalibConstants in your DB? ";
  }

  // === amplitude computation ===
  int leadingSample = -1;
  if (detid.subdetId() == EcalEndcap) {
    leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
  } else {
    leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
  }

  if (leadingSample == 4) {  // saturation on the expected max sample
    uncalibRecHit = EcalUncalibratedRecHit((*itdg).id(), 4095 * 12, 0, 0, 0);
    uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kSaturated);
    // do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
    uncalibRecHit.setChi2(0);
  } else if (leadingSample >= 0) {  // saturation on other samples: cannot extrapolate from the fourth one
    double pedestal = 0.;
    double gainratio = 1.;
    int gainId = ((EcalDataFrame)(*itdg)).sample(5).gainId();

    if (gainId == 0 || gainId == 3) {
      pedestal = aped->mean_x1;
      gainratio = aGain->gain6Over1() * aGain->gain12Over6();
    } else if (gainId == 1) {
      pedestal = aped->mean_x12;
      gainratio = 1.;
    } else if (gainId == 2) {
      pedestal = aped->mean_x6;
      gainratio = aGain->gain12Over6();
    }
    double amplitude = ((double)(((EcalDataFrame)(*itdg)).sample(5).adc()) - pedestal) * gainratio;
    uncalibRecHit = EcalUncalibratedRecHit((*itdg).id(), amplitude, 0, 0, 0);
    uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kSaturated);
    // do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
    uncalibRecHit.setChi2(0);
  } else {
    // weights method
    EcalTBWeights::EcalTDCId tdcid(1);
    EcalTBWeights::EcalTBWeightMap const& wgtsMap = wgts_->getMap();
    EcalTBWeights::EcalTBWeightMap::const_iterator wit;
    wit = wgtsMap.find(std::make_pair(*gid, tdcid));
    if (wit == wgtsMap.end()) {
      edm::LogError("EcalUncalibRecHitError")
          << "No weights found for EcalGroupId: " << gid->id() << " and  EcalTDCId: " << tdcid
          << "\n  skipping digi with id: " << detid.rawId();

      return false;
    }
    const EcalWeightSet& wset = wit->second;  // this is the EcalWeightSet

    const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
    const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();

    weights[0] = &mat1;
    weights[1] = &mat2;

    // get uncalibrated recHit from weights
    if (detid.subdetId() == EcalEndcap) {
      uncalibRecHit = weightsMethod_endcap_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEEShape);
    } else {
      uncalibRecHit = weightsMethod_barrel_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEBShape);
    }

    // === time computation ===
    // ratio method
    float const clockToNsConstant = 25.;
    if (detid.subdetId() == EcalEndcap) {
      ratioMethod_endcap_.init(*itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios);
      ratioMethod_endcap_.computeTime(EEtimeFitParameters_, EEtimeFitLimits_, EEamplitudeFitParameters_);
      ratioMethod_endcap_.computeAmplitude(EEamplitudeFitParameters_);
      EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh = ratioMethod_endcap_.getCalculatedRecHit();
      double theTimeCorrectionEE = timeCorrection(
          uncalibRecHit.amplitude(), timeCorrBias_->EETimeCorrAmplitudeBins, timeCorrBias_->EETimeCorrShiftBins);

      uncalibRecHit.setJitter(crh.timeMax - 5 + theTimeCorrectionEE);
      uncalibRecHit.setJitterError(
          std::sqrt(pow(crh.timeError, 2) + std::pow(EEtimeConstantTerm_, 2) / std::pow(clockToNsConstant, 2)));
      // consider flagging as kOutOfTime only if above noise
      if (uncalibRecHit.amplitude() > pedRMSVec[0] * amplitudeThreshEE_) {
        float outOfTimeThreshP = outOfTimeThreshG12pEE_;
        float outOfTimeThreshM = outOfTimeThreshG12mEE_;
        // determine if gain has switched away from gainId==1 (x12 gain)
        // and determine cuts (number of 'sigmas') to ose for kOutOfTime
        // >3k ADC is necessasry condition for gain switch to occur
        if (uncalibRecHit.amplitude() > 3000.) {
          for (int iSample = 0; iSample < EEDataFrame::MAXSAMPLES; iSample++) {
            int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
            if (GainId != 1) {
              outOfTimeThreshP = outOfTimeThreshG61pEE_;
              outOfTimeThreshM = outOfTimeThreshG61mEE_;
              break;
            }
          }
        }
        float correctedTime = (crh.timeMax - 5) * clockToNsConstant + itimeconst + offsetTime;
        float cterm = EEtimeConstantTerm_;
        float sigmaped = pedRMSVec[0];  // approx for lower gains
        float nterm = EEtimeNconst_ * sigmaped / uncalibRecHit.amplitude();
        float sigmat = std::sqrt(nterm * nterm + cterm * cterm);
        if ((correctedTime > sigmat * outOfTimeThreshP) || (correctedTime < (-1. * sigmat * outOfTimeThreshM))) {
          uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kOutOfTime);
        }
      }

    } else {
      ratioMethod_barrel_.init(*itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios);
      ratioMethod_barrel_.fixMGPAslew(*itdg);
      ratioMethod_barrel_.computeTime(EBtimeFitParameters_, EBtimeFitLimits_, EBamplitudeFitParameters_);
      ratioMethod_barrel_.computeAmplitude(EBamplitudeFitParameters_);
      EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh = ratioMethod_barrel_.getCalculatedRecHit();

      double theTimeCorrectionEB = timeCorrection(
          uncalibRecHit.amplitude(), timeCorrBias_->EBTimeCorrAmplitudeBins, timeCorrBias_->EBTimeCorrShiftBins);

      uncalibRecHit.setJitter(crh.timeMax - 5 + theTimeCorrectionEB);

      uncalibRecHit.setJitterError(
          std::sqrt(std::pow(crh.timeError, 2) + std::pow(EBtimeConstantTerm_, 2) / std::pow(clockToNsConstant, 2)));
      // consider flagging as kOutOfTime only if above noise
      if (uncalibRecHit.amplitude() > pedRMSVec[0] * amplitudeThreshEB_) {
        float outOfTimeThreshP = outOfTimeThreshG12pEB_;
        float outOfTimeThreshM = outOfTimeThreshG12mEB_;
        // determine if gain has switched away from gainId==1 (x12 gain)
        // and determine cuts (number of 'sigmas') to ose for kOutOfTime
        // >3k ADC is necessasry condition for gain switch to occur
        if (uncalibRecHit.amplitude() > 3000.) {
          for (int iSample = 0; iSample < EBDataFrame::MAXSAMPLES; iSample++) {
            int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
            if (GainId != 1) {
              outOfTimeThreshP = outOfTimeThreshG61pEB_;
              outOfTimeThreshM = outOfTimeThreshG61mEB_;
              break;
            }
          }
        }
        float correctedTime = (crh.timeMax - 5) * clockToNsConstant + itimeconst + offsetTime;
        float cterm = EBtimeConstantTerm_;
        float sigmaped = pedRMSVec[0];  // approx for lower gains
        float nterm = EBtimeNconst_ * sigmaped / uncalibRecHit.amplitude();
        float sigmat = std::sqrt(nterm * nterm + cterm * cterm);
        if ((correctedTime > sigmat * outOfTimeThreshP) || (correctedTime < (-1. * sigmat * outOfTimeThreshM))) {
          uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kOutOfTime);
        }
      }
    }

    // === chi2express ===
    if (detid.subdetId() == EcalEndcap) {
      double amplitude = uncalibRecHit.amplitude();
      double amplitudeOutOfTime = 0.;
      double jitter = uncalibRecHit.jitter();

      EcalUncalibRecHitRecChi2Algo<EEDataFrame> chi2expressEE_(*itdg,
                                                               amplitude,
                                                               (itimeconst + offsetTime),
                                                               amplitudeOutOfTime,
                                                               jitter,
                                                               pedVec,
                                                               pedRMSVec,
                                                               gainRatios,
                                                               testbeamEEShape,
                                                               EEchi2Parameters_);
      double chi2 = chi2expressEE_.chi2();
      uncalibRecHit.setChi2(chi2);

      if (kPoorRecoFlagEE_) {
        if (chi2 > chi2ThreshEE_) {
          // first check if all samples are ok, if not don't use chi2 to flag
          bool samplesok = true;
          for (int sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
            if (!sampleMask_->useSampleEE(sample)) {
              samplesok = false;
              break;
            }
          }
          if (samplesok)
            uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
        }
      }

    } else {
      double amplitude = uncalibRecHit.amplitude();
      double amplitudeOutOfTime = 0.;
      double jitter = uncalibRecHit.jitter();

      EcalUncalibRecHitRecChi2Algo<EBDataFrame> chi2expressEB_(*itdg,
                                                               amplitude,
                                                               (itimeconst + offsetTime),
                                                               amplitudeOutOfTime,
                                                               jitter,
                                                               pedVec,
                                                               pedRMSVec,
                                                               gainRatios,
                                                               testbeamEBShape,
                                                               EBchi2Parameters_);
      double chi2 = chi2expressEB_.chi2();
      uncalibRecHit.setChi2(chi2);

      if (kPoorRecoFlagEB_) {
        if (chi2 > chi2ThreshEB_) {
          // first check if all samples are ok, if not don't use chi2 to flag
          bool samplesok = true;
          for (int sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
            if (!sampleMask_->useSampleEB(sample)) {
              samplesok = false;
              break;
            }
          }
          if (samplesok)
            uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
        }
      }
    }
  }

  // set flags if gain switch has occurred
  if (((EcalDataFrame)(*itdg)).hasSwitchToGain6())
    uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain6);
  if (((EcalDataFrame)(*itdg)).hasSwitchToGain1())
    uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain1);

  // put the recHit in the collection
  if (detid.subdetId() == EcalEndcap) {
    result.push_back(uncalibRecHit);
  } else {
    result.push_back(uncalibRecHit);
  }

  return true;
}

edm::ParameterSetDescription EcalUncalibRecHitWorkerGlobal::getAlgoDescription() {
  edm::ParameterSetDescription psd;
  psd.addNode(
      edm::ParameterDescription<std::vector<double>>(
          "eePulseShape", {5.2e-05, -5.26e-05, 6.66e-05, 0.1168, 0.7575, 1.0, 0.8876, 0.6732, 0.4741, 0.3194}, true) and
      edm::ParameterDescription<std::vector<double>>(
          "EBtimeFitParameters",
          {-2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621},
          true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain61pEB", 5, true) and
      edm::ParameterDescription<double>("amplitudeThresholdEE", 10, true) and
      edm::ParameterDescription<double>("EBtimeConstantTerm", 0.6, true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain61pEE", 1000, true) and
      edm::ParameterDescription<double>("ebSpikeThreshold", 1.042, true) and
      edm::ParameterDescription<double>("EBtimeNconst", 28.5, true) and
      edm::ParameterDescription<bool>("kPoorRecoFlagEB", true, true) and
      edm::ParameterDescription<std::vector<double>>(
          "ebPulseShape", {5.2e-05, -5.26e-05, 6.66e-05, 0.1168, 0.7575, 1.0, 0.8876, 0.6732, 0.4741, 0.3194}, true) and
      edm::ParameterDescription<double>("EBtimeFitLimits_Lower", 0.2, true) and
      edm::ParameterDescription<bool>("kPoorRecoFlagEE", false, true) and
      edm::ParameterDescription<double>("chi2ThreshEB_", 36.0, true) and
      edm::ParameterDescription<std::vector<double>>(
          "EEtimeFitParameters",
          {-2.390548, 3.553628, -17.62341, 67.67538, -133.213, 140.7432, -75.41106, 16.20277},
          true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain61mEE", 1000, true) and
      edm::ParameterDescription<std::vector<double>>("EEchi2Parameters", {2.122, 0.022, 2.122, 0.022}, true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain12mEE", 1000, true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain12mEB", 5, true) and
      edm::ParameterDescription<double>("EEtimeFitLimits_Upper", 1.4, true) and
      edm::ParameterDescription<double>("EEtimeFitLimits_Lower", 0.2, true) and
      edm::ParameterDescription<std::vector<double>>("EEamplitudeFitParameters", {1.89, 1.4}, true) and
      edm::ParameterDescription<std::vector<double>>("EBamplitudeFitParameters", {1.138, 1.652}, true) and
      edm::ParameterDescription<double>("amplitudeThresholdEB", 10, true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain12pEE", 1000, true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain12pEB", 5, true) and
      edm::ParameterDescription<double>("EEtimeNconst", 31.8, true) and
      edm::ParameterDescription<double>("outOfTimeThresholdGain61mEB", 5, true) and
      edm::ParameterDescription<std::vector<double>>("EBchi2Parameters", {2.122, 0.022, 2.122, 0.022}, true) and
      edm::ParameterDescription<double>("EEtimeConstantTerm", 1.0, true) and
      edm::ParameterDescription<double>("chi2ThreshEE_", 95.0, true) and
      edm::ParameterDescription<double>("EBtimeFitLimits_Upper", 1.4, true));

  return psd;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerGlobal, "EcalUncalibRecHitWorkerGlobal");
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitFillDescriptionWorkerFactory,
                  EcalUncalibRecHitWorkerGlobal,
                  "EcalUncalibRecHitWorkerGlobal");
