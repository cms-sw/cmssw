/** \class EcalUncalibRecHitRecGlobalAlgo
 *  Template used to compute amplitude, pedestal using a weights method
 *                           time using a ratio method
 *                           chi2 using express method
 *
 *  \author R. Bruneliere - A. Zabi
 */

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRatioMethodAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecChi2Algo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimeWeightsAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimingCCAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"

class EcalUncalibRecHitWorkerMultiFit final : public EcalUncalibRecHitWorkerBaseClass {
public:
  EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerMultiFit(){};

private:
  void set(const edm::EventSetup& es) override;
  void set(const edm::Event& evt) override;
  void run(const edm::Event& evt, const EcalDigiCollection& digis, EcalUncalibratedRecHitCollection& result) override;

public:
  edm::ParameterSetDescription getAlgoDescription() override;

private:
  edm::ESHandle<EcalPedestals> peds;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedsToken_;
  edm::ESHandle<EcalGainRatios> gains;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainsToken_;
  edm::ESHandle<EcalSamplesCorrelation> noisecovariances;
  edm::ESGetToken<EcalSamplesCorrelation, EcalSamplesCorrelationRcd> noiseConvariancesToken_;
  edm::ESHandle<EcalPulseShapes> pulseshapes;
  edm::ESGetToken<EcalPulseShapes, EcalPulseShapesRcd> pulseShapesToken_;
  edm::ESHandle<EcalPulseCovariances> pulsecovariances;
  edm::ESGetToken<EcalPulseCovariances, EcalPulseCovariancesRcd> pulseConvariancesToken_;

  double timeCorrection(float ampli, const std::vector<float>& amplitudeBins, const std::vector<float>& shiftBins);

  const SampleMatrix& noisecor(bool barrel, int gain) const { return noisecors_[barrel ? 1 : 0][gain]; }
  const SampleMatrixGainArray& noisecor(bool barrel) const { return noisecors_[barrel ? 1 : 0]; }

  // multifit method
  std::array<SampleMatrixGainArray, 2> noisecors_;
  BXVector activeBX;
  bool ampErrorCalculation_;
  bool useLumiInfoRunHeader_;
  EcalUncalibRecHitMultiFitAlgo multiFitMethod_;

  int bunchSpacingManual_;
  edm::EDGetTokenT<unsigned int> bunchSpacing_;

  // determine which of the samples must actually be used by ECAL local reco
  edm::ESHandle<EcalSampleMask> sampleMaskHand_;
  edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> sampleMaskToken_;

  // time algorithm to be used to set the jitter and its uncertainty
  enum TimeAlgo { noMethod, ratioMethod, weightsMethod, crossCorrelationMethod };
  TimeAlgo timealgo_ = noMethod;

  // time weights method
  edm::ESHandle<EcalWeightXtalGroups> grps;
  edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> grpsToken_;
  edm::ESHandle<EcalTBWeights> wgts;
  edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> wgtsToken_;
  const EcalWeightSet::EcalWeightMatrix* weights[2];
  EcalUncalibRecHitTimeWeightsAlgo<EBDataFrame> weightsMethod_barrel_;
  EcalUncalibRecHitTimeWeightsAlgo<EEDataFrame> weightsMethod_endcap_;
  bool doPrefitEB_;
  bool doPrefitEE_;
  double prefitMaxChiSqEB_;
  double prefitMaxChiSqEE_;
  bool dynamicPedestalsEB_;
  bool dynamicPedestalsEE_;
  bool mitigateBadSamplesEB_;
  bool mitigateBadSamplesEE_;
  bool gainSwitchUseMaxSampleEB_;
  bool gainSwitchUseMaxSampleEE_;
  bool selectiveBadSampleCriteriaEB_;
  bool selectiveBadSampleCriteriaEE_;
  double addPedestalUncertaintyEB_;
  double addPedestalUncertaintyEE_;
  bool simplifiedNoiseModelForGainSwitch_;

  // ratio method
  std::vector<double> EBtimeFitParameters_;
  std::vector<double> EEtimeFitParameters_;
  std::vector<double> EBamplitudeFitParameters_;
  std::vector<double> EEamplitudeFitParameters_;
  std::pair<double, double> EBtimeFitLimits_;
  std::pair<double, double> EEtimeFitLimits_;

  EcalUncalibRecHitRatioMethodAlgo<EBDataFrame> ratioMethod_barrel_;
  EcalUncalibRecHitRatioMethodAlgo<EEDataFrame> ratioMethod_endcap_;

  double EBtimeConstantTerm_;
  double EEtimeConstantTerm_;
  double EBtimeNconst_;
  double EEtimeNconst_;
  double outOfTimeThreshG12pEB_;
  double outOfTimeThreshG12mEB_;
  double outOfTimeThreshG61pEB_;
  double outOfTimeThreshG61mEB_;
  double outOfTimeThreshG12pEE_;
  double outOfTimeThreshG12mEE_;
  double outOfTimeThreshG61pEE_;
  double outOfTimeThreshG61mEE_;
  double amplitudeThreshEB_;
  double amplitudeThreshEE_;
  double ebSpikeThresh_;

  edm::ESHandle<EcalTimeBiasCorrections> timeCorrBias_;
  edm::ESGetToken<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd> timeCorrBiasToken_;

  edm::ESHandle<EcalTimeCalibConstants> itime;
  edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> itimeToken_;
  edm::ESHandle<EcalTimeOffsetConstant> offtime;
  edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> offtimeToken_;
  std::vector<double> ebPulseShape_;
  std::vector<double> eePulseShape_;

  // chi2 thresholds for flags settings
  bool kPoorRecoFlagEB_;
  bool kPoorRecoFlagEE_;
  double chi2ThreshEB_;
  double chi2ThreshEE_;

  //Timing Cross Correlation Algo
  std::unique_ptr<EcalUncalibRecHitTimingCCAlgo> computeCC_;
  double CCminTimeToBeLateMin_;
  double CCminTimeToBeLateMax_;
  double CCTimeShiftWrtRations_;
  double CCtargetTimePrecision_;
  double CCtargetTimePrecisionForDelayedPulses_;
};

EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet& ps, edm::ConsumesCollector& c)
    : EcalUncalibRecHitWorkerBaseClass(ps, c) {
  // get the BX for the pulses to be activated
  std::vector<int32_t> activeBXs = ps.getParameter<std::vector<int32_t>>("activeBXs");
  activeBX.resize(activeBXs.size());
  for (unsigned int ibx = 0; ibx < activeBXs.size(); ++ibx) {
    activeBX.coeffRef(ibx) = activeBXs[ibx];
  }

  // uncertainty calculation (CPU intensive)
  ampErrorCalculation_ = ps.getParameter<bool>("ampErrorCalculation");
  useLumiInfoRunHeader_ = ps.getParameter<bool>("useLumiInfoRunHeader");

  if (useLumiInfoRunHeader_) {
    bunchSpacing_ = c.consumes<unsigned int>(edm::InputTag("bunchSpacingProducer"));
    bunchSpacingManual_ = 0;
  } else {
    bunchSpacingManual_ = ps.getParameter<int>("bunchSpacing");
  }

  doPrefitEB_ = ps.getParameter<bool>("doPrefitEB");
  doPrefitEE_ = ps.getParameter<bool>("doPrefitEE");

  prefitMaxChiSqEB_ = ps.getParameter<double>("prefitMaxChiSqEB");
  prefitMaxChiSqEE_ = ps.getParameter<double>("prefitMaxChiSqEE");

  dynamicPedestalsEB_ = ps.getParameter<bool>("dynamicPedestalsEB");
  dynamicPedestalsEE_ = ps.getParameter<bool>("dynamicPedestalsEE");
  mitigateBadSamplesEB_ = ps.getParameter<bool>("mitigateBadSamplesEB");
  mitigateBadSamplesEE_ = ps.getParameter<bool>("mitigateBadSamplesEE");
  gainSwitchUseMaxSampleEB_ = ps.getParameter<bool>("gainSwitchUseMaxSampleEB");
  gainSwitchUseMaxSampleEE_ = ps.getParameter<bool>("gainSwitchUseMaxSampleEE");
  selectiveBadSampleCriteriaEB_ = ps.getParameter<bool>("selectiveBadSampleCriteriaEB");
  selectiveBadSampleCriteriaEE_ = ps.getParameter<bool>("selectiveBadSampleCriteriaEE");
  addPedestalUncertaintyEB_ = ps.getParameter<double>("addPedestalUncertaintyEB");
  addPedestalUncertaintyEE_ = ps.getParameter<double>("addPedestalUncertaintyEE");
  simplifiedNoiseModelForGainSwitch_ = ps.getParameter<bool>("simplifiedNoiseModelForGainSwitch");
  pedsToken_ = c.esConsumes<EcalPedestals, EcalPedestalsRcd>();
  gainsToken_ = c.esConsumes<EcalGainRatios, EcalGainRatiosRcd>();
  noiseConvariancesToken_ = c.esConsumes<EcalSamplesCorrelation, EcalSamplesCorrelationRcd>();
  pulseShapesToken_ = c.esConsumes<EcalPulseShapes, EcalPulseShapesRcd>();
  pulseConvariancesToken_ = c.esConsumes<EcalPulseCovariances, EcalPulseCovariancesRcd>();
  sampleMaskToken_ = c.esConsumes<EcalSampleMask, EcalSampleMaskRcd>();
  grpsToken_ = c.esConsumes<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd>();
  wgtsToken_ = c.esConsumes<EcalTBWeights, EcalTBWeightsRcd>();
  timeCorrBiasToken_ = c.esConsumes<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd>();
  itimeToken_ = c.esConsumes<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd>();
  offtimeToken_ = c.esConsumes<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd>();

  // algorithm to be used for timing
  auto const& timeAlgoName = ps.getParameter<std::string>("timealgo");
  if (timeAlgoName == "RatioMethod")
    timealgo_ = ratioMethod;
  else if (timeAlgoName == "WeightsMethod")
    timealgo_ = weightsMethod;
  else if (timeAlgoName == "crossCorrelationMethod") {
    timealgo_ = crossCorrelationMethod;
    double startTime = ps.getParameter<double>("crossCorrelationStartTime");
    double stopTime = ps.getParameter<double>("crossCorrelationStopTime");
    CCtargetTimePrecision_ = ps.getParameter<double>("crossCorrelationTargetTimePrecision");
    CCtargetTimePrecisionForDelayedPulses_ =
        ps.getParameter<double>("crossCorrelationTargetTimePrecisionForDelayedPulses");
    CCminTimeToBeLateMin_ = ps.getParameter<double>("crossCorrelationMinTimeToBeLateMin") / ecalcctiming::clockToNS;
    CCminTimeToBeLateMax_ = ps.getParameter<double>("crossCorrelationMinTimeToBeLateMax") / ecalcctiming::clockToNS;
    CCTimeShiftWrtRations_ = ps.getParameter<double>("crossCorrelationTimeShiftWrtRations");
    computeCC_ = std::make_unique<EcalUncalibRecHitTimingCCAlgo>(startTime, stopTime);
  } else if (timeAlgoName != "None")
    edm::LogError("EcalUncalibRecHitError") << "No time estimation algorithm defined";

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
  EEtimeConstantTerm_ = ps.getParameter<double>("EEtimeConstantTerm");
  EBtimeNconst_ = ps.getParameter<double>("EBtimeNconst");
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
}

void EcalUncalibRecHitWorkerMultiFit::set(const edm::EventSetup& es) {
  // common setup
  gains = es.getHandle(gainsToken_);
  peds = es.getHandle(pedsToken_);

  // for the multifit method
  if (!ampErrorCalculation_)
    multiFitMethod_.disableErrorCalculation();
  noisecovariances = es.getHandle(noiseConvariancesToken_);
  pulseshapes = es.getHandle(pulseShapesToken_);
  pulsecovariances = es.getHandle(pulseConvariancesToken_);

  // weights parameters for the time
  grps = es.getHandle(grpsToken_);
  wgts = es.getHandle(wgtsToken_);

  // which of the samples need be used
  sampleMaskHand_ = es.getHandle(sampleMaskToken_);

  // for the ratio method
  itime = es.getHandle(itimeToken_);
  offtime = es.getHandle(offtimeToken_);

  // for the time correction methods
  timeCorrBias_ = es.getHandle(timeCorrBiasToken_);

  int nnoise = SampleVector::RowsAtCompileTime;
  SampleMatrix& noisecorEBg12 = noisecors_[1][0];
  SampleMatrix& noisecorEBg6 = noisecors_[1][1];
  SampleMatrix& noisecorEBg1 = noisecors_[1][2];
  SampleMatrix& noisecorEEg12 = noisecors_[0][0];
  SampleMatrix& noisecorEEg6 = noisecors_[0][1];
  SampleMatrix& noisecorEEg1 = noisecors_[0][2];

  for (int i = 0; i < nnoise; ++i) {
    for (int j = 0; j < nnoise; ++j) {
      int vidx = std::abs(j - i);
      noisecorEBg12(i, j) = noisecovariances->EBG12SamplesCorrelation[vidx];
      noisecorEEg12(i, j) = noisecovariances->EEG12SamplesCorrelation[vidx];
      noisecorEBg6(i, j) = noisecovariances->EBG6SamplesCorrelation[vidx];
      noisecorEEg6(i, j) = noisecovariances->EEG6SamplesCorrelation[vidx];
      noisecorEBg1(i, j) = noisecovariances->EBG1SamplesCorrelation[vidx];
      noisecorEEg1(i, j) = noisecovariances->EEG1SamplesCorrelation[vidx];
    }
  }
}

void EcalUncalibRecHitWorkerMultiFit::set(const edm::Event& evt) {
  unsigned int bunchspacing = 450;

  if (useLumiInfoRunHeader_) {
    edm::Handle<unsigned int> bunchSpacingH;
    evt.getByToken(bunchSpacing_, bunchSpacingH);
    bunchspacing = *bunchSpacingH;
  } else {
    bunchspacing = bunchSpacingManual_;
  }

  if (useLumiInfoRunHeader_ || bunchSpacingManual_ > 0) {
    if (bunchspacing == 25) {
      activeBX.resize(10);
      activeBX << -5, -4, -3, -2, -1, 0, 1, 2, 3, 4;
    } else {
      //50ns configuration otherwise (also for no pileup)
      activeBX.resize(5);
      activeBX << -4, -2, 0, 2, 4;
    }
  }
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
double EcalUncalibRecHitWorkerMultiFit::timeCorrection(float ampli,
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

  // FIXME? what about a binary search?
  int myBin = -1;
  for (int bin = 0; bin < (int)amplitudeBins.size(); bin++) {
    if (ampli > amplitudeBins[bin]) {
      myBin = bin;
    } else {
      break;
    }
  }

  if (myBin == -1) {
    theCorrection = shiftBins[0];
  } else if (myBin == ((int)(amplitudeBins.size() - 1))) {
    theCorrection = shiftBins[myBin];
  } else {
    // interpolate linearly between two assingned points
    theCorrection = (shiftBins[myBin + 1] - shiftBins[myBin]);
    theCorrection *= (((double)ampli) - amplitudeBins[myBin]) / (amplitudeBins[myBin + 1] - amplitudeBins[myBin]);
    theCorrection += shiftBins[myBin];
  }

  // convert ns into clocks
  constexpr double inv25 = 1. / 25.;
  return theCorrection * inv25;
}

void EcalUncalibRecHitWorkerMultiFit::run(const edm::Event& evt,
                                          const EcalDigiCollection& digis,
                                          EcalUncalibratedRecHitCollection& result) {
  if (digis.empty())
    return;

  // assume all digis come from the same subdetector (either barrel or endcap)
  DetId detid(digis.begin()->id());
  bool barrel = (detid.subdetId() == EcalBarrel);

  multiFitMethod_.setSimplifiedNoiseModelForGainSwitch(simplifiedNoiseModelForGainSwitch_);
  if (barrel) {
    multiFitMethod_.setDoPrefit(doPrefitEB_);
    multiFitMethod_.setPrefitMaxChiSq(prefitMaxChiSqEB_);
    multiFitMethod_.setDynamicPedestals(dynamicPedestalsEB_);
    multiFitMethod_.setMitigateBadSamples(mitigateBadSamplesEB_);
    multiFitMethod_.setGainSwitchUseMaxSample(gainSwitchUseMaxSampleEB_);
    multiFitMethod_.setSelectiveBadSampleCriteria(selectiveBadSampleCriteriaEB_);
    multiFitMethod_.setAddPedestalUncertainty(addPedestalUncertaintyEB_);
  } else {
    multiFitMethod_.setDoPrefit(doPrefitEE_);
    multiFitMethod_.setPrefitMaxChiSq(prefitMaxChiSqEE_);
    multiFitMethod_.setDynamicPedestals(dynamicPedestalsEE_);
    multiFitMethod_.setMitigateBadSamples(mitigateBadSamplesEE_);
    multiFitMethod_.setGainSwitchUseMaxSample(gainSwitchUseMaxSampleEE_);
    multiFitMethod_.setSelectiveBadSampleCriteria(selectiveBadSampleCriteriaEE_);
    multiFitMethod_.setAddPedestalUncertainty(addPedestalUncertaintyEE_);
  }

  FullSampleVector fullpulse(FullSampleVector::Zero());
  FullSampleMatrix fullpulsecov(FullSampleMatrix::Zero());

  result.reserve(result.size() + digis.size());
  for (auto itdg = digis.begin(); itdg != digis.end(); ++itdg) {
    DetId detid(itdg->id());

    const EcalSampleMask* sampleMask_ = sampleMaskHand_.product();

    // intelligence for recHit computation
    float offsetTime = 0;

    const EcalPedestals::Item* aped = nullptr;
    const EcalMGPAGainRatio* aGain = nullptr;
    const EcalXtalGroupId* gid = nullptr;
    const EcalPulseShapes::Item* aPulse = nullptr;
    const EcalPulseCovariances::Item* aPulseCov = nullptr;

    if (barrel) {
      unsigned int hashedIndex = EBDetId(detid).hashedIndex();
      aped = &peds->barrel(hashedIndex);
      aGain = &gains->barrel(hashedIndex);
      gid = &grps->barrel(hashedIndex);
      aPulse = &pulseshapes->barrel(hashedIndex);
      aPulseCov = &pulsecovariances->barrel(hashedIndex);
      offsetTime = offtime->getEBValue();
    } else {
      unsigned int hashedIndex = EEDetId(detid).hashedIndex();
      aped = &peds->endcap(hashedIndex);
      aGain = &gains->endcap(hashedIndex);
      gid = &grps->endcap(hashedIndex);
      aPulse = &pulseshapes->endcap(hashedIndex);
      aPulseCov = &pulsecovariances->endcap(hashedIndex);
      offsetTime = offtime->getEEValue();
    }

    double pedVec[3] = {aped->mean_x12, aped->mean_x6, aped->mean_x1};
    double pedRMSVec[3] = {aped->rms_x12, aped->rms_x6, aped->rms_x1};
    double gainRatios[3] = {1., aGain->gain12Over6(), aGain->gain6Over1() * aGain->gain12Over6()};

    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; ++i)
      fullpulse(i + 7) = aPulse->pdfval[i];

    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; i++)
      for (int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; j++)
        fullpulsecov(i + 7, j + 7) = aPulseCov->covval[i][j];

    // compute the right bin of the pulse shape using time calibration constants
    EcalTimeCalibConstantMap::const_iterator it = itime->find(detid);
    EcalTimeCalibConstant itimeconst = 0;
    if (it != itime->end()) {
      itimeconst = (*it);
    } else {
      edm::LogError("EcalRecHitError") << "No time intercalib const found for xtal " << detid.rawId()
                                       << "! something wrong with EcalTimeCalibConstants in your DB? ";
    }

    int lastSampleBeforeSaturation = -2;
    for (unsigned int iSample = 0; iSample < EcalDataFrame::MAXSAMPLES; iSample++) {
      if (((EcalDataFrame)(*itdg)).sample(iSample).gainId() == 0) {
        lastSampleBeforeSaturation = iSample - 1;
        break;
      }
    }

    // === amplitude computation ===

    if (lastSampleBeforeSaturation == 4) {  // saturation on the expected max sample
      result.emplace_back((*itdg).id(), 4095 * 12, 0, 0, 0);
      auto& uncalibRecHit = result.back();
      uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kSaturated);
      // do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
      uncalibRecHit.setChi2(0);
    } else if (lastSampleBeforeSaturation >=
               -1) {  // saturation on other samples: cannot extrapolate from the fourth one
      int gainId = ((EcalDataFrame)(*itdg)).sample(5).gainId();
      if (gainId == 0)
        gainId = 3;
      auto pedestal = pedVec[gainId - 1];
      auto gainratio = gainRatios[gainId - 1];
      double amplitude = ((double)(((EcalDataFrame)(*itdg)).sample(5).adc()) - pedestal) * gainratio;
      result.emplace_back((*itdg).id(), amplitude, 0, 0, 0);
      auto& uncalibRecHit = result.back();
      uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kSaturated);
      // do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
      uncalibRecHit.setChi2(0);
    } else {
      // multifit
      const SampleMatrixGainArray& noisecors = noisecor(barrel);

      result.push_back(multiFitMethod_.makeRecHit(*itdg, aped, aGain, noisecors, fullpulse, fullpulsecov, activeBX));
      auto& uncalibRecHit = result.back();

      // === time computation ===
      if (timealgo_ == ratioMethod) {
        // ratio method
        constexpr float clockToNsConstant = 25.;
        constexpr float invClockToNs = 1. / clockToNsConstant;
        if (not barrel) {
          ratioMethod_endcap_.init(*itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios);
          ratioMethod_endcap_.computeTime(EEtimeFitParameters_, EEtimeFitLimits_, EEamplitudeFitParameters_);
          ratioMethod_endcap_.computeAmplitude(EEamplitudeFitParameters_);
          EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh =
              ratioMethod_endcap_.getCalculatedRecHit();
          double theTimeCorrectionEE = timeCorrection(
              uncalibRecHit.amplitude(), timeCorrBias_->EETimeCorrAmplitudeBins, timeCorrBias_->EETimeCorrShiftBins);

          uncalibRecHit.setJitter(crh.timeMax - 5 + theTimeCorrectionEE);
          uncalibRecHit.setJitterError(
              std::sqrt(std::pow(crh.timeError, 2) + std::pow(EEtimeConstantTerm_ * invClockToNs, 2)));

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
            if ((correctedTime > sigmat * outOfTimeThreshP) || (correctedTime < -sigmat * outOfTimeThreshM)) {
              uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kOutOfTime);
            }
          }

        } else {
          ratioMethod_barrel_.init(*itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios);
          ratioMethod_barrel_.fixMGPAslew(*itdg);
          ratioMethod_barrel_.computeTime(EBtimeFitParameters_, EBtimeFitLimits_, EBamplitudeFitParameters_);
          ratioMethod_barrel_.computeAmplitude(EBamplitudeFitParameters_);
          EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh =
              ratioMethod_barrel_.getCalculatedRecHit();

          double theTimeCorrectionEB = timeCorrection(
              uncalibRecHit.amplitude(), timeCorrBias_->EBTimeCorrAmplitudeBins, timeCorrBias_->EBTimeCorrShiftBins);

          uncalibRecHit.setJitter(crh.timeMax - 5 + theTimeCorrectionEB);
          uncalibRecHit.setJitterError(std::hypot(crh.timeError, EBtimeConstantTerm_ / clockToNsConstant));

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
            if ((correctedTime > sigmat * outOfTimeThreshP) || (correctedTime < -sigmat * outOfTimeThreshM)) {
              uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kOutOfTime);
            }
          }
        }
      } else if (timealgo_ == weightsMethod) {
        //  weights method on the PU subtracted pulse shape
        std::vector<double> amplitudes;
        for (unsigned int ibx = 0; ibx < activeBX.size(); ++ibx)
          amplitudes.push_back(uncalibRecHit.outOfTimeAmplitude(ibx));

        EcalTBWeights::EcalTDCId tdcid(1);
        EcalTBWeights::EcalTBWeightMap const& wgtsMap = wgts->getMap();
        EcalTBWeights::EcalTBWeightMap::const_iterator wit;
        wit = wgtsMap.find(std::make_pair(*gid, tdcid));
        if (wit == wgtsMap.end()) {
          edm::LogError("EcalUncalibRecHitError")
              << "No weights found for EcalGroupId: " << gid->id() << " and  EcalTDCId: " << tdcid
              << "\n  skipping digi with id: " << detid.rawId();
          result.pop_back();
          continue;
        }
        const EcalWeightSet& wset = wit->second;  // this is the EcalWeightSet

        const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
        const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();

        weights[0] = &mat1;
        weights[1] = &mat2;

        double timerh;
        if (detid.subdetId() == EcalEndcap) {
          timerh = weightsMethod_endcap_.time(*itdg, amplitudes, aped, aGain, fullpulse, weights);
        } else {
          timerh = weightsMethod_barrel_.time(*itdg, amplitudes, aped, aGain, fullpulse, weights);
        }
        uncalibRecHit.setJitter(timerh);
        uncalibRecHit.setJitterError(0.);  // not computed with weights

      } else if (timealgo_ == crossCorrelationMethod) {
        std::vector<double> amplitudes(activeBX.size());
        for (unsigned int ibx = 0; ibx < activeBX.size(); ++ibx)
          amplitudes[ibx] = uncalibRecHit.outOfTimeAmplitude(ibx);

        float jitter =
            computeCC_->computeTimeCC(*itdg, amplitudes, aped, aGain, fullpulse, CCtargetTimePrecision_, true) +
            CCTimeShiftWrtRations_ / ecalcctiming::clockToNS;
        float noCorrectedJitter =
            computeCC_->computeTimeCC(
                *itdg, amplitudes, aped, aGain, fullpulse, CCtargetTimePrecisionForDelayedPulses_, false) +
            CCTimeShiftWrtRations_ / ecalcctiming::clockToNS;

        uncalibRecHit.setJitter(jitter);
        uncalibRecHit.setNonCorrectedTime(jitter, noCorrectedJitter);

        float retreivedNonCorrectedTime = uncalibRecHit.nonCorrectedTime();
        float noCorrectedTime = ecalcctiming::clockToNS * noCorrectedJitter;
        if (retreivedNonCorrectedTime > -29.0 && std::abs(retreivedNonCorrectedTime - noCorrectedTime) > 0.05) {
          edm::LogError("EcalUncalibRecHitError") << "Problem with noCorrectedJitter: true value:" << noCorrectedTime
                                                  << "\t received: " << retreivedNonCorrectedTime << std::endl;
        }  //<<>>if (abs(retreivedNonCorrectedTime - noCorrectedJitter)>1);

        // consider flagging as kOutOfTime only if above noise
        float threshold, cterm, timeNconst;
        float timeThrP = 0.;
        float timeThrM = 0.;
        if (barrel) {
          threshold = pedRMSVec[0] * amplitudeThreshEB_;
          cterm = EBtimeConstantTerm_;
          timeNconst = EBtimeNconst_;
          timeThrP = outOfTimeThreshG12pEB_;
          timeThrM = outOfTimeThreshG12mEB_;
          if (uncalibRecHit.amplitude() > 3000.) {  // Gain switch
            for (int iSample = 0; iSample < EBDataFrame::MAXSAMPLES; iSample++) {
              int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
              if (GainId != 1) {
                timeThrP = outOfTimeThreshG61pEB_;
                timeThrM = outOfTimeThreshG61mEB_;
                break;
              }
            }
          }
        } else {  //EndCap
          threshold = pedRMSVec[0] * amplitudeThreshEE_;
          cterm = EEtimeConstantTerm_;
          timeNconst = EEtimeNconst_;
          timeThrP = outOfTimeThreshG12pEE_;
          timeThrM = outOfTimeThreshG12mEE_;
          if (uncalibRecHit.amplitude() > 3000.) {  // Gain switch
            for (int iSample = 0; iSample < EEDataFrame::MAXSAMPLES; iSample++) {
              int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
              if (GainId != 1) {
                timeThrP = outOfTimeThreshG61pEE_;
                timeThrM = outOfTimeThreshG61mEE_;
                break;
              }
            }
          }
        }
        if (uncalibRecHit.amplitude() > threshold) {
          float correctedTime = noCorrectedJitter * ecalcctiming::clockToNS + itimeconst + offsetTime;
          float sigmaped = pedRMSVec[0];  // approx for lower gains
          float nterm = timeNconst * sigmaped / uncalibRecHit.amplitude();
          float sigmat = std::sqrt(nterm * nterm + cterm * cterm);
          if ((correctedTime > sigmat * timeThrP) || (correctedTime < -sigmat * timeThrM))
            uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kOutOfTime);
        }

      } else {  // no time method;
        uncalibRecHit.setJitter(0.);
        uncalibRecHit.setJitterError(0.);
      }
    }

    // set flags if gain switch has occurred
    auto& uncalibRecHit = result.back();
    if (((EcalDataFrame)(*itdg)).hasSwitchToGain6())
      uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain6);
    if (((EcalDataFrame)(*itdg)).hasSwitchToGain1())
      uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain1);
  }
}

edm::ParameterSetDescription EcalUncalibRecHitWorkerMultiFit::getAlgoDescription() {
  edm::ParameterSetDescription psd;
  psd.addNode(edm::ParameterDescription<std::vector<int>>("activeBXs", {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4}, true) and
              edm::ParameterDescription<bool>("ampErrorCalculation", true, true) and
              edm::ParameterDescription<bool>("useLumiInfoRunHeader", true, true) and
              edm::ParameterDescription<int>("bunchSpacing", 0, true) and
              edm::ParameterDescription<bool>("doPrefitEB", false, true) and
              edm::ParameterDescription<bool>("doPrefitEE", false, true) and
              edm::ParameterDescription<double>("prefitMaxChiSqEB", 25., true) and
              edm::ParameterDescription<double>("prefitMaxChiSqEE", 10., true) and
              edm::ParameterDescription<bool>("dynamicPedestalsEB", false, true) and
              edm::ParameterDescription<bool>("dynamicPedestalsEE", false, true) and
              edm::ParameterDescription<bool>("mitigateBadSamplesEB", false, true) and
              edm::ParameterDescription<bool>("mitigateBadSamplesEE", false, true) and
              edm::ParameterDescription<bool>("gainSwitchUseMaxSampleEB", false, true) and
              edm::ParameterDescription<bool>("gainSwitchUseMaxSampleEE", false, true) and
              edm::ParameterDescription<bool>("selectiveBadSampleCriteriaEB", false, true) and
              edm::ParameterDescription<bool>("selectiveBadSampleCriteriaEE", false, true) and
              edm::ParameterDescription<double>("addPedestalUncertaintyEB", 0., true) and
              edm::ParameterDescription<double>("addPedestalUncertaintyEE", 0., true) and
              edm::ParameterDescription<bool>("simplifiedNoiseModelForGainSwitch", true, true) and
              edm::ParameterDescription<std::string>("timealgo", "crossCorrelationMethod", true) and
              edm::ParameterDescription<std::vector<double>>("EBtimeFitParameters",
                                                             {-2.015452e+00,
                                                              3.130702e+00,
                                                              -1.234730e+01,
                                                              4.188921e+01,
                                                              -8.283944e+01,
                                                              9.101147e+01,
                                                              -5.035761e+01,
                                                              1.105621e+01},
                                                             true) and
              edm::ParameterDescription<std::vector<double>>("EEtimeFitParameters",
                                                             {-2.390548e+00,
                                                              3.553628e+00,
                                                              -1.762341e+01,
                                                              6.767538e+01,
                                                              -1.332130e+02,
                                                              1.407432e+02,
                                                              -7.541106e+01,
                                                              1.620277e+01},
                                                             true) and
              edm::ParameterDescription<std::vector<double>>("EBamplitudeFitParameters", {1.138, 1.652}, true) and
              edm::ParameterDescription<std::vector<double>>("EEamplitudeFitParameters", {1.890, 1.400}, true) and
              edm::ParameterDescription<double>("EBtimeFitLimits_Lower", 0.2, true) and
              edm::ParameterDescription<double>("EBtimeFitLimits_Upper", 1.4, true) and
              edm::ParameterDescription<double>("EEtimeFitLimits_Lower", 0.2, true) and
              edm::ParameterDescription<double>("EEtimeFitLimits_Upper", 1.4, true) and
              edm::ParameterDescription<double>("EBtimeConstantTerm", .6, true) and
              edm::ParameterDescription<double>("EEtimeConstantTerm", 1.0, true) and
              edm::ParameterDescription<double>("EBtimeNconst", 28.5, true) and
              edm::ParameterDescription<double>("EEtimeNconst", 31.8, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain12pEB", 2.5, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain12mEB", 2.5, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain61pEB", 2.5, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain61mEB", 2.5, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain12pEE", 1000, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain12mEE", 1000, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain61pEE", 1000, true) and
              edm::ParameterDescription<double>("outOfTimeThresholdGain61mEE", 1000, true) and
              edm::ParameterDescription<double>("amplitudeThresholdEB", 10, true) and
              edm::ParameterDescription<double>("amplitudeThresholdEE", 10, true) and
              edm::ParameterDescription<double>("crossCorrelationStartTime", -25.0, true) and
              edm::ParameterDescription<double>("crossCorrelationStopTime", 25.0, true) and
              edm::ParameterDescription<double>("crossCorrelationTargetTimePrecision", 0.01, true) and
              edm::ParameterDescription<double>("crossCorrelationTargetTimePrecisionForDelayedPulses", 0.05, true) and
              edm::ParameterDescription<double>("crossCorrelationTimeShiftWrtRations", 0., true) and
              edm::ParameterDescription<double>("crossCorrelationMinTimeToBeLateMin", 2., true) and
              edm::ParameterDescription<double>("crossCorrelationMinTimeToBeLateMax", 5., true));

  return psd;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMultiFit, "EcalUncalibRecHitWorkerMultiFit");
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitFillDescriptionWorkerFactory,
                  EcalUncalibRecHitWorkerMultiFit,
                  "EcalUncalibRecHitWorkerMultiFit");
