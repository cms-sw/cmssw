
#ifndef PulseShapeFitOOTPileupCorrection_h
#define PulseShapeFitOOTPileupCorrection_h 1

#include <typeinfo>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include "TFile.h"
#include "TTree.h"

#include <TMinuit.h>
#include "TFitterMinuit.h"

#include <TH1F.h>
#include "Minuit2/FCNBase.h"
#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Functor.h"

#include "RecoLocalCalo/HcalRecAlgos/src/HybridMinimizer.h"

namespace FitterFuncs{
  
   class PulseShapeFunctor {
      public:
     PulseShapeFunctor(const HcalPulseShapes::Shape& pulse,bool iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,
		       double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
		       double iNoise);
         ~PulseShapeFunctor();
         double EvalSinglePulse(const std::vector<double>& pars) const;
         double EvalDoublePulse(const std::vector<double>& pars) const;
         double EvalTriplePulse(const std::vector<double>& pars) const;
      private:
         std::array<float,256> pulse_hist;
         std::vector<float> acc25nsVec, diff25nsItvlVec;
         std::vector<float> accVarLenIdxZEROVec, diffVarItvlIdxZEROVec;
         std::vector<float> accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec;
	 bool pedestalConstraint_;
	 bool timeConstraint_;
	 bool addPulseJitter_;
	 double pulseJitter_;
	 double timeMean_;
	 double timeSig_;
	 double pedMean_;
	 double pedSig_;
	 double noise_;
   };
   
}

class PulseShapeFitOOTPileupCorrection
{
public:
    PulseShapeFitOOTPileupCorrection();
    ~PulseShapeFitOOTPileupCorrection();

    void apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const;
    void setPUParams(bool   iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,
		     double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
		     double iNoise,double iTMin,double iTMax,
		     double its3Chi2,double its4Chi2,double its345Chi2,HcalTimeSlew::BiasSetting slewFlavor);
    
    void setPulseShapeTemplate  (const HcalPulseShapes::Shape& ps);
    void resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps);

    void setChargeThreshold(double chargeThrInput){ chargeThreshold_ = chargeThrInput; }

private:
    int pulseShapeFit(const double * energyArr, const double * pedenArr, const double *chargeArr, const double *pedArr, const double tsTOTen, std::vector<double> &fitParsVec) const;
    void fit(int iFit,float &timevalfit,float &chargevalfit,float &pedvalfit,float &chi2,bool &fitStatus,double &iTSMax,const double  &iTSTOTen,int (&iBX)[3]) const;
    PSFitter::HybridMinimizer * hybridfitter;
    int cntsetPulseShape;
    std::array<double,10> iniTimesArr;
    double chargeThreshold_;
    ROOT::Math::Functor *spfunctor_;
    ROOT::Math::Functor *dpfunctor_;
    ROOT::Math::Functor *tpfunctor_;
    int TSMin_;
    int TSMax_;
    double ts4Chi2_;
    double ts3Chi2_;
    double ts345Chi2_;
    bool pedestalConstraint_;
    bool timeConstraint_;
    bool addPulseJitter_;
    double pulseJitter_;
    double timeMean_;
    double timeSig_;
    double pedMean_;
    double pedSig_;
    double noise_;    
    HcalTimeSlew::BiasSetting slewFlavor_;    
};

#endif // PulseShapeFitOOTPileupCorrection_h
