#ifndef RecoLocalCalo_HcalRecAlgos_PulseShapeFitOOTPileupCorrection_h
#define RecoLocalCalo_HcalRecAlgos_PulseShapeFitOOTPileupCorrection_h

#include <typeinfo>

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"

#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include <TMinuit.h>

#include <TH1F.h>
#include "Minuit2/FCNBase.h"
#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Functor.h"

#include "RecoLocalCalo/HcalRecAlgos/src/HybridMinimizer.h"

class HcalTimeSlew;

class PulseShapeFitOOTPileupCorrection {
public:
  PulseShapeFitOOTPileupCorrection();
  ~PulseShapeFitOOTPileupCorrection();

  void phase1Apply(const HBHEChannelInfo &channelData,
                   float &reconstructedEnergy,
                   float &reconstructedTime,
                   bool &useTriple,
                   float &chi2) const;

  void setPUParams(bool iPedestalConstraint,
                   bool iTimeConstraint,
                   bool iAddPulseJitter,
                   bool iApplyTimeSlew,
                   double iTS4Min,
                   const std::vector<double> &iTS4Max,
                   double iPulseJitter,
                   double iTimeMean,
                   double iTimeSigHPD,
                   double iTimeSigSiPM,
                   double iPedMean,
                   double iTMin,
                   double iTMax,
                   const std::vector<double> &its4Chi2,
                   HcalTimeSlew::BiasSetting slewFlavor,
                   int iFitTimes);

  const HcalPulseShapes::Shape *currentPulseShape_ = nullptr;
  const HcalTimeSlew *hcalTimeSlewDelay_ = nullptr;
  double tsDelay1GeV_ = 0;

  void setPulseShapeTemplate(const HcalPulseShapes::Shape &ps,
                             bool isHPD,
                             unsigned nSamples,
                             const HcalTimeSlew *hcalTimeSlewDelay);
  void resetPulseShapeTemplate(const HcalPulseShapes::Shape &ps, unsigned nSamples);

private:
  int pulseShapeFit(const double *energyArr,
                    const double *pedenArr,
                    const double *chargeArr,
                    const double *pedArr,
                    const double *gainArr,
                    const double tsTOTen,
                    std::vector<float> &fitParsVec,
                    const double *ADCnoise,
                    unsigned int soi) const;
  void fit(int iFit,
           float &timevalfit,
           float &chargevalfit,
           float &pedvalfit,
           float &chi2,
           bool &fitStatus,
           double &iTSMax,
           const double &iTSTOTen,
           double *iEnArr,
           unsigned (&iBX)[3]) const;

  PSFitter::HybridMinimizer *hybridfitter;
  int cntsetPulseShape;
  std::array<double, hcal::constants::maxSamples> iniTimesArr;
  double chargeThreshold_;
  int fitTimes_;

  std::unique_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;
  std::unique_ptr<ROOT::Math::Functor> spfunctor_;
  std::unique_ptr<ROOT::Math::Functor> dpfunctor_;
  std::unique_ptr<ROOT::Math::Functor> tpfunctor_;
  int TSMin_;
  int TSMax_;
  mutable double ts4Chi2_;
  std::vector<double> vts4Chi2_;
  bool pedestalConstraint_;
  bool timeConstraint_;
  bool addPulseJitter_;
  bool unConstrainedFit_;
  bool applyTimeSlew_;
  double ts4Min_;
  mutable double ts4Max_;
  std::vector<double> vts4Max_;
  double pulseJitter_;
  double timeMean_;
  double timeSig_;
  double timeSigHPD_;
  double timeSigSiPM_;
  double pedMean_;
  double pedSig_;
  HcalTimeSlew::BiasSetting slewFlavor_;

  bool isCurrentChannelHPD_;
};

#endif  // PulseShapeFitOOTPileupCorrection_h
