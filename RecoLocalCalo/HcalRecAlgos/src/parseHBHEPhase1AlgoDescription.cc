#include <cassert>

#include "RecoLocalCalo/HcalRecAlgos/interface/parseHBHEPhase1AlgoDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"

// Phase 1 HBHE reco algorithm headers
#include "RecoLocalCalo/HcalRecAlgos/interface/SimpleHBHEPhase1Algo.h"


static std::unique_ptr<PulseShapeFitOOTPileupCorrection>
parseHBHEMethod2Description(const edm::ParameterSet& conf)
{
    const bool iPedestalConstraint = conf.getParameter<bool>  ("applyPedConstraint");
    const bool iTimeConstraint =     conf.getParameter<bool>  ("applyTimeConstraint");
    const bool iAddPulseJitter =     conf.getParameter<bool>  ("applyPulseJitter");
    const bool iApplyTimeSlew =      conf.getParameter<bool>  ("applyTimeSlew");
    const double iTS4Min =           conf.getParameter<double>("ts4Min");
    const std::vector<double> iTS4Max =           conf.getParameter<std::vector<double>>("ts4Max");
    const double iPulseJitter =      conf.getParameter<double>("pulseJitter");
    const double iTimeMean =         conf.getParameter<double>("meanTime");
    const double iTimeSigHPD =       conf.getParameter<double>("timeSigmaHPD");
    const double iTimeSigSiPM =      conf.getParameter<double>("timeSigmaSiPM");
    const double iPedMean =          conf.getParameter<double>("meanPed");
    const double iPedSigHPD =        conf.getParameter<double>("pedSigmaHPD");
    const double iPedSigSiPM =       conf.getParameter<double>("pedSigmaSiPM");
    const double iNoiseHPD =         conf.getParameter<double>("noiseHPD");
    const double iNoiseSiPM =        conf.getParameter<double>("noiseSiPM");
    const double iTMin =             conf.getParameter<double>("timeMin");
    const double iTMax =             conf.getParameter<double>("timeMax");
    const std::vector<double> its4Chi2 =           conf.getParameter<std::vector<double>>("ts4chi2");
    const int iFitTimes =            conf.getParameter<int>   ("fitTimes");

    if (iPedestalConstraint) assert(iPedSigHPD);
    if (iPedestalConstraint) assert(iPedSigSiPM);
    if (iTimeConstraint) assert(iTimeSigHPD);
    if (iTimeConstraint) assert(iTimeSigSiPM);

    std::unique_ptr<PulseShapeFitOOTPileupCorrection> corr =
        std::make_unique<PulseShapeFitOOTPileupCorrection>();

    corr->setPUParams(iPedestalConstraint, iTimeConstraint, iAddPulseJitter,
                      iApplyTimeSlew, iTS4Min, iTS4Max,
                      iPulseJitter,
		      iTimeMean, iTimeSigHPD, iTimeSigSiPM,
		      iPedMean, iPedSigHPD, iPedSigSiPM,
                      iNoiseHPD, iNoiseSiPM,
		      iTMin, iTMax, its4Chi2,
                      HcalTimeSlew::Medium, iFitTimes);

    return corr;
}


static std::unique_ptr<HcalDeterministicFit>
parseHBHEMethod3Description(const edm::ParameterSet& conf)
{
    const bool iApplyTimeSlew  =  conf.getParameter<bool>  ("applyTimeSlewM3");
    const float iPedSubThreshold =  conf.getParameter<double>("pedestalUpperLimit");
    const int iTimeSlewParsType  =  conf.getParameter<int>   ("timeSlewParsType");
    const double irespCorrM3 =     conf.getParameter<double>("respCorrM3");
    const std::vector<double>& iTimeSlewPars =
                     conf.getParameter<std::vector<double> >("timeSlewPars");

    PedestalSub pedSubFxn;
    pedSubFxn.init(0, iPedSubThreshold, 0.0);

    std::unique_ptr<HcalDeterministicFit> fit = std::make_unique<HcalDeterministicFit>();
    fit->init( (HcalTimeSlew::ParaSource)iTimeSlewParsType,
	       HcalTimeSlew::Medium, iApplyTimeSlew,
	       pedSubFxn, iTimeSlewPars, irespCorrM3);
    return fit;
}


std::unique_ptr<AbsHBHEPhase1Algo>
parseHBHEPhase1AlgoDescription(const edm::ParameterSet& ps)
{
    std::unique_ptr<AbsHBHEPhase1Algo> algo;

    const std::string& className = ps.getParameter<std::string>("Class");

    if (className == "SimpleHBHEPhase1Algo")
    {
        std::unique_ptr<PulseShapeFitOOTPileupCorrection> m2;
        if (ps.getParameter<bool>("useM2"))
            m2 = parseHBHEMethod2Description(ps);

        std::unique_ptr<HcalDeterministicFit> detFit;
        if (ps.getParameter<bool>("useM3"))
            detFit = parseHBHEMethod3Description(ps);

        algo = std::unique_ptr<AbsHBHEPhase1Algo>(
            new SimpleHBHEPhase1Algo(ps.getParameter<int>   ("firstSampleShift"),
                                     ps.getParameter<int>   ("samplesToAdd"),
                                     ps.getParameter<double>("correctionPhaseNS"),
                                     ps.getParameter<double>("tdcTimeShift"),
                                     ps.getParameter<bool>  ("correctForPhaseContainment"),
                                     std::move(m2), std::move(detFit))
            );
    }

    return algo;
}
