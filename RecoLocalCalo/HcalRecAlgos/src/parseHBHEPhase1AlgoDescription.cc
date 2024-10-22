#include <cassert>

#include "RecoLocalCalo/HcalRecAlgos/interface/parseHBHEPhase1AlgoDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"

// Phase 1 HBHE reco algorithm headers
#include "RecoLocalCalo/HcalRecAlgos/interface/SimpleHBHEPhase1Algo.h"

static std::unique_ptr<MahiFit> parseHBHEMahiDescription(const edm::ParameterSet& conf) {
  const bool iDynamicPed = conf.getParameter<bool>("dynamicPed");
  const double iTS4Thresh = conf.getParameter<double>("ts4Thresh");
  const double chiSqSwitch = conf.getParameter<double>("chiSqSwitch");

  const bool iApplyTimeSlew = conf.getParameter<bool>("applyTimeSlew");

  const bool iCalculateArrivalTime = conf.getParameter<bool>("calculateArrivalTime");
  const int iTimeAlgo = conf.getParameter<int>("timeAlgo");
  const double iThEnergeticPulses = conf.getParameter<double>("thEnergeticPulses");
  const double iMeanTime = conf.getParameter<double>("meanTime");
  const double iTimeSigmaHPD = conf.getParameter<double>("timeSigmaHPD");
  const double iTimeSigmaSiPM = conf.getParameter<double>("timeSigmaSiPM");

  const std::vector<int> iActiveBXs = conf.getParameter<std::vector<int>>("activeBXs");
  const int iNMaxItersMin = conf.getParameter<int>("nMaxItersMin");
  const int iNMaxItersNNLS = conf.getParameter<int>("nMaxItersNNLS");
  const double iDeltaChiSqThresh = conf.getParameter<double>("deltaChiSqThresh");
  const double iNnlsThresh = conf.getParameter<double>("nnlsThresh");

  std::unique_ptr<MahiFit> corr = std::make_unique<MahiFit>();

  corr->setParameters(iDynamicPed,
                      iTS4Thresh,
                      chiSqSwitch,
                      iApplyTimeSlew,
                      HcalTimeSlew::Medium,
                      iCalculateArrivalTime,
                      iTimeAlgo,
                      iThEnergeticPulses,
                      iMeanTime,
                      iTimeSigmaHPD,
                      iTimeSigmaSiPM,
                      iActiveBXs,
                      iNMaxItersMin,
                      iNMaxItersNNLS,
                      iDeltaChiSqThresh,
                      iNnlsThresh);

  return corr;
}

static std::unique_ptr<PulseShapeFitOOTPileupCorrection> parseHBHEMethod2Description(const edm::ParameterSet& conf) {
  const bool iPedestalConstraint = conf.getParameter<bool>("applyPedConstraint");
  const bool iTimeConstraint = conf.getParameter<bool>("applyTimeConstraint");
  const bool iAddPulseJitter = conf.getParameter<bool>("applyPulseJitter");
  const bool iApplyTimeSlew = conf.getParameter<bool>("applyTimeSlew");
  const double iTS4Min = conf.getParameter<double>("ts4Min");
  const std::vector<double> iTS4Max = conf.getParameter<std::vector<double>>("ts4Max");
  const double iPulseJitter = conf.getParameter<double>("pulseJitter");
  const double iTimeMean = conf.getParameter<double>("meanTime");
  const double iTimeSigHPD = conf.getParameter<double>("timeSigmaHPD");
  const double iTimeSigSiPM = conf.getParameter<double>("timeSigmaSiPM");
  const double iPedMean = conf.getParameter<double>("meanPed");
  const double iTMin = conf.getParameter<double>("timeMin");
  const double iTMax = conf.getParameter<double>("timeMax");
  const std::vector<double> its4Chi2 = conf.getParameter<std::vector<double>>("ts4chi2");
  const int iFitTimes = conf.getParameter<int>("fitTimes");

  if (iTimeConstraint)
    assert(iTimeSigHPD);
  if (iTimeConstraint)
    assert(iTimeSigSiPM);

  std::unique_ptr<PulseShapeFitOOTPileupCorrection> corr = std::make_unique<PulseShapeFitOOTPileupCorrection>();

  corr->setPUParams(iPedestalConstraint,
                    iTimeConstraint,
                    iAddPulseJitter,
                    iApplyTimeSlew,
                    iTS4Min,
                    iTS4Max,
                    iPulseJitter,
                    iTimeMean,
                    iTimeSigHPD,
                    iTimeSigSiPM,
                    iPedMean,
                    iTMin,
                    iTMax,
                    its4Chi2,
                    HcalTimeSlew::Medium,
                    iFitTimes);

  return corr;
}

static std::unique_ptr<HcalDeterministicFit> parseHBHEMethod3Description(const edm::ParameterSet& conf) {
  const bool iApplyTimeSlew = conf.getParameter<bool>("applyTimeSlewM3");
  const int iTimeSlewParsType = conf.getParameter<int>("timeSlewParsType");
  const double irespCorrM3 = conf.getParameter<double>("respCorrM3");

  std::unique_ptr<HcalDeterministicFit> fit = std::make_unique<HcalDeterministicFit>();

  fit->init((HcalTimeSlew::ParaSource)iTimeSlewParsType, HcalTimeSlew::Medium, iApplyTimeSlew, irespCorrM3);

  return fit;
}

std::unique_ptr<AbsHBHEPhase1Algo> parseHBHEPhase1AlgoDescription(const edm::ParameterSet& ps,
                                                                  edm::ConsumesCollector iC) {
  std::unique_ptr<AbsHBHEPhase1Algo> algo;

  const std::string& className = ps.getParameter<std::string>("Class");

  if (className == "SimpleHBHEPhase1Algo") {
    std::unique_ptr<MahiFit> mahi;
    std::unique_ptr<PulseShapeFitOOTPileupCorrection> m2;
    std::unique_ptr<HcalDeterministicFit> detFit;

    // only run Mahi OR Method 2 but not both
    if (ps.getParameter<bool>("useMahi") && ps.getParameter<bool>("useM2")) {
      throw cms::Exception("ConfigurationError")
          << "SimpleHBHEPhase1Algo does not allow both Mahi and Method 2 to be turned on together.";
    }
    if (ps.getParameter<bool>("useMahi"))
      mahi = parseHBHEMahiDescription(ps);
    if (ps.getParameter<bool>("useM2"))
      m2 = parseHBHEMethod2Description(ps);
    if (ps.getParameter<bool>("useM3"))
      detFit = parseHBHEMethod3Description(ps);

    algo = std::make_unique<SimpleHBHEPhase1Algo>(ps.getParameter<int>("firstSampleShift"),
                                                  ps.getParameter<int>("samplesToAdd"),
                                                  ps.getParameter<double>("correctionPhaseNS"),
                                                  ps.getParameter<double>("tdcTimeShift"),
                                                  ps.getParameter<bool>("correctForPhaseContainment"),
                                                  ps.getParameter<bool>("applyLegacyHBMCorrection"),
                                                  ps.getParameter<bool>("applyFixPCC"),
                                                  std::move(m2),
                                                  std::move(detFit),
                                                  std::move(mahi),
                                                  iC);
  }

  return algo;
}

edm::ParameterSetDescription fillDescriptionForParseHBHEPhase1Algo() {
  edm::ParameterSetDescription desc;

  desc.setAllowAnything();
  desc.add<std::string>("Class", "SimpleHBHEPhase1Algo");
  desc.add<bool>("useM2", false);
  desc.add<bool>("useM3", true);
  desc.add<bool>("useMahi", true);
  desc.add<int>("firstSampleShift", 0);
  desc.add<int>("samplesToAdd", 2);
  desc.add<double>("correctionPhaseNS", 6.0);
  desc.add<double>("tdcTimeShift", 0.0);
  desc.add<bool>("correctForPhaseContainment", true);
  desc.add<bool>("applyLegacyHBMCorrection", true);
  desc.add<bool>("calculateArrivalTime", false);
  desc.add<int>("timeAlgo", 1);
  desc.add<double>("thEnergeticPulses", 5.);
  desc.add<bool>("applyFixPCC", false);

  return desc;
}
