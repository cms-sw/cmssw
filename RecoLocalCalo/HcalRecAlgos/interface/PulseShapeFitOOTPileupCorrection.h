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

#include <TH1F.h>
#include "Minuit2/FCNBase.h"
#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Functor.h"

#include "RecoLocalCalo/HcalRecAlgos/src/HybridMinimizer.h"

namespace FitterFuncs{
  
   class PulseShapeFunctor {
      public:
     PulseShapeFunctor(const HcalPulseShapes::Shape& pulse,bool iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,bool iAddTimeSlew,
		       double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
		       double iNoise);
     ~PulseShapeFunctor();
     
     double EvalSinglePulse(const std::vector<double>& pars);
     double EvalDoublePulse(const std::vector<double>& pars);
     double EvalTriplePulse(const std::vector<double>& pars);
     
     void setDefaultcntNANinfit(){ cntNANinfit =0; }
     int getcntNANinfit(){ return cntNANinfit; }
     
     void setpsFitx(double *x ){ for(int i=0; i<10; ++i) psFit_x[i] = x[i]; }
     void setpsFity(double *y ){ for(int i=0; i<10; ++i) psFit_y[i] = y[i]; }
     void setpsFiterry (double *erry  ){ for(int i=0; i<10; ++i) psFit_erry  [i] = erry [i]; }
     void setpsFiterry2(double *erry2 ){ for(int i=0; i<10; ++i) psFit_erry2 [i] = erry2[i]; }
     void setpsFitslew (double *slew  ){ for(int i=0; i<10; ++i) {psFit_slew [i] = slew [i]; } }
     double sigma(double ifC);
     double singlePulseShapeFunc( const double *x );
     double doublePulseShapeFunc( const double *x );
     double triplePulseShapeFunc( const double *x );
     
   private:
     std::array<float,256> pulse_hist;
     
     int cntNANinfit;
     std::vector<float> acc25nsVec, diff25nsItvlVec;
     std::vector<float> accVarLenIdxZEROVec, diffVarItvlIdxZEROVec;
     std::vector<float> accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec;
     std::array<float,10> funcHPDShape(const double &pulseTime, const double &pulseHeight,const double &slew);
     double psFit_x[10], psFit_y[10], psFit_erry[10], psFit_erry2[10], psFit_slew[10];
     
     bool pedestalConstraint_;
     bool timeConstraint_;
     bool addPulseJitter_;
     bool unConstrainedFit_;
     double pulseJitter_;
     double timeMean_;
     double timeSig_;
     double pedMean_;
     double pedSig_;
     double noise_;
     double timeShift_;
   };
   
}

class PulseShapeFitOOTPileupCorrection
{
public:
    PulseShapeFitOOTPileupCorrection();
    ~PulseShapeFitOOTPileupCorrection();

    void apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const;
    void setPUParams(bool   iPedestalConstraint, bool iTimeConstraint,bool iAddPulseJitter,bool iUnConstrainedFit,bool iApplyTimeSlew,
		     double iTS4Min,double iPulseJitter,double iTimeMean,double iTimeSig,double iPedMean,double iPedSig,
		     double iNoise,double iTMin,double iTMax,
		     double its3Chi2,double its4Chi2,double its345Chi2,double iChargeThreshold,HcalTimeSlew::BiasSetting slewFlavor);
    
    void setPulseShapeTemplate  (const HcalPulseShapes::Shape& ps);
    void resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps);
    void setChargeThreshold(double chargeThrInput){ chargeThreshold_ = chargeThrInput; }

private:
    int pulseShapeFit(const double * energyArr, const double * pedenArr, const double *chargeArr, 
		      const double *pedArr, const double tsTOTen, std::vector<double> &fitParsVec) const;
    void fit(int iFit,float &timevalfit,float &chargevalfit,float &pedvalfit,float &chi2,bool &fitStatus,double &iTSMax,
	     const double  &iTSTOTen,double *iEnArr,int (&iBX)[3]) const;

    PSFitter::HybridMinimizer * hybridfitter;
    int cntsetPulseShape;
    std::array<double,10> iniTimesArr;
    double chargeThreshold_;

    std::auto_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;
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
    bool unConstrainedFit_;
    bool applyTimeSlew_;
    double ts4Min_;
    double pulseJitter_;
    double timeMean_;
    double timeSig_;
    double pedMean_;
    double pedSig_;
    double noise_;    
    HcalTimeSlew::BiasSetting slewFlavor_;    
};

#endif // PulseShapeFitOOTPileupCorrection_h
