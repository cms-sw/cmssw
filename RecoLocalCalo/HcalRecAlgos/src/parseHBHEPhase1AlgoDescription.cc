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
    const bool iUnConstrainedFit =   conf.getParameter<bool>  ("applyUnconstrainedFit");
    const bool iApplyTimeSlew =      conf.getParameter<bool>  ("applyTimeSlew");
    const double iTS4Min =           conf.getParameter<double>("ts4Min");
    const double iTS4Max =           conf.getParameter<double>("ts4Max");
    const double iPulseJitter =      conf.getParameter<double>("pulseJitter");
    const double iTimeMean =         conf.getParameter<double>("meanTime");
    const double iTimeSig =          conf.getParameter<double>("timeSigma");
    const double iPedMean =          conf.getParameter<double>("meanPed");
    const double iPedSig =           conf.getParameter<double>("pedSigma");
    const double iNoise =            conf.getParameter<double>("noise");
    const double iTMin =             conf.getParameter<double>("timeMin");
    const double iTMax =             conf.getParameter<double>("timeMax");
    const double its4Chi2 =          conf.getParameter<double>("ts4chi2");
    const double iChargeThreshold =  conf.getParameter<double>("chargeMax"); //For the unconstrained Fit
    const int iFitTimes =            conf.getParameter<int>   ("fitTimes");

    if (iPedestalConstraint) assert(iPedSig);
    if (iTimeConstraint) assert(iTimeSig);

    std::unique_ptr<PulseShapeFitOOTPileupCorrection> corr =
        std::make_unique<PulseShapeFitOOTPileupCorrection>();
    corr->setPUParams(iPedestalConstraint, iTimeConstraint, iAddPulseJitter,
                      iUnConstrainedFit, iApplyTimeSlew, iTS4Min, iTS4Max,
                      iPulseJitter, iTimeMean, iTimeSig, iPedMean, iPedSig,
                      iNoise, iTMin, iTMax, its4Chi2,
                      iChargeThreshold, HcalTimeSlew::Medium, iFitTimes);
    return corr;
}


static std::unique_ptr<HcalDeterministicFit>
parseHBHEMethod3Description(const edm::ParameterSet& conf)
{
    const int iPedSubMethod =      conf.getParameter<int>   ("pedestalSubtractionType");
    const float iPedSubThreshold = conf.getParameter<double>("pedestalUpperLimit");
    const int iTimeSlewParsType =  conf.getParameter<int>   ("timeSlewParsType");
    const double irespCorrM3 =     conf.getParameter<double>("respCorrM3");
    const std::vector<double>& iTimeSlewPars =
                     conf.getParameter<std::vector<double> >("timeSlewPars");

    PedestalSub pedSubFxn;
    pedSubFxn.init(((PedestalSub::Method)iPedSubMethod), 0, iPedSubThreshold, 0.0);

    std::unique_ptr<HcalDeterministicFit> fit = std::make_unique<HcalDeterministicFit>();
    fit->init( (HcalTimeSlew::ParaSource)iTimeSlewParsType,
	       HcalTimeSlew::Medium,
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
                                     std::move(m2), std::move(detFit))
            );
    }

    return algo;
}
