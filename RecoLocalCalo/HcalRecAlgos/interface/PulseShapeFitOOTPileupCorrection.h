#ifndef PulseShapeFitOOTPileupCorrection_h
#define PulseShapeFitOOTPileupCorrection_h 1

#include <typeinfo>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"

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
         PulseShapeFunctor(const HcalPulseShapes::Shape& pulse);
         ~PulseShapeFunctor();
         double EvalSinglePulse(const std::vector<double>& pars) const;
         double EvalDoublePulse(const std::vector<double>& pars) const;
      private:
         std::array<float,256> pulse_hist;
         std::vector<float> acc25nsVec, diff25nsItvlVec;
         std::vector<float> accVarLenIdxZEROVec, diffVarItvlIdxZEROVec;
         std::vector<float> accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec;
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
};

#endif // PulseShapeFitOOTPileupCorrection_h
