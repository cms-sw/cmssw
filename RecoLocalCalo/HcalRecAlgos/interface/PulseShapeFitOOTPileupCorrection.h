#ifndef PulseShapeFitOOTPileupCorrection_h
#define PulseShapeFitOOTPileupCorrection_h 1

#include <typeinfo>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

#include <TMinuit.h>

#include <TH1F.h>
#include "Minuit2/FCNBase.h"
#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Functor.h"

#include "RecoLocalCalo/HcalRecAlgos/src/HybridMinimizer.h"

namespace FitterFuncs{
  
   class PulseShapeFunctor {
      public:
         PulseShapeFunctor(const HcalPulseShapes::Shape& pulse);
         ~PulseShapeFunctor();

         double EvalSinglePulse(const std::vector<double>& pars);
         double EvalDoublePulse(const std::vector<double>& pars);

         void setDefaultcntNANinfit(){ cntNANinfit =0; }
         int getcntNANinfit(){ return cntNANinfit; }

         void setpsFitx(double *x ){ for(int i=0; i<10; ++i) psFit_x[i] = x[i]; }
         void setpsFity(double *y ){ for(int i=0; i<10; ++i) psFit_y[i] = y[i]; }
         void setpsFiterry(double *erry ){ for(int i=0; i<10; ++i) psFit_erry[i] = erry[i]; }

         double singlePulseShapeFunc( const double *x );
         double doublePulseShapeFunc( const double *x );
      private:
         std::array<float,256> pulse_hist;

         int cntNANinfit;

         std::vector<float> acc25nsVec, diff25nsItvlVec;
         std::vector<float> accVarLenIdxZEROVec, diffVarItvlIdxZEROVec;
         std::vector<float> accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec;

         std::array<float,10> funcHPDShape(const std::vector<double>& pars);
         std::array<float,10> func_DoublePulse_HPDShape(const std::vector<double>& pars);

         double psFit_x[10], psFit_y[10], psFit_erry[10];
   };
   
}

class PulseShapeFitOOTPileupCorrection
{
public:
    PulseShapeFitOOTPileupCorrection();
    ~PulseShapeFitOOTPileupCorrection();

    void apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const;

    void setPulseShapeTemplate(const HcalPulseShapes::Shape& ps);
    void resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps);
    void setChargeThreshold(double chargeThrInput){ chargeThreshold_ = chargeThrInput; }

private:

    int pulseShapeFit(const double * energyArr, const double * pedenArr, const double *chargeArr, const double *pedArr, const double tsTOTen, std::vector<double> &fitParsVec) const;

    PSFitter::HybridMinimizer * hybridfitter;

    int cntsetPulseShape;

    std::array<double,10> iniTimesArr;

    double chargeThreshold_;

    std::auto_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;

};

#endif // PulseShapeFitOOTPileupCorrection_h
