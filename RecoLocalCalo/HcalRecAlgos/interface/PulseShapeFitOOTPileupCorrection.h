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

namespace FitterFuncs{
  
   class SinglePulseShapeFunctor {
      public:
         SinglePulseShapeFunctor(const HcalPulseShapes::Shape& pulse);
         ~SinglePulseShapeFunctor();
         double operator()(const std::vector<double>& pars) const;
      private:
         std::array<float,256> pulse_hist;
   };
   
   class DoublePulseShapeFunctor {
      public:
         DoublePulseShapeFunctor(const HcalPulseShapes::Shape& pulse);
         ~DoublePulseShapeFunctor();
         double operator()(const std::vector<double> & pars) const;
      private:
         std::array<float,256> pulse_hist;
   };
   
   // because minuit owns the function you pass it?
   template<typename PSF>
   class PulseShapeFCN : public ROOT::Minuit2::FCNBase {
      public:
          inline PulseShapeFCN(const PSF* t) { psf_ = t; }
          inline double operator()(const std::vector<double>& pars) const {
             return (*psf_)(pars);
          }
          inline double Up() const { return 1.; }
      private:
         const PSF* psf_;
   };
   
}

class PulseShapeFitOOTPileupCorrection
{
public:
    PulseShapeFitOOTPileupCorrection();
    inline ~PulseShapeFitOOTPileupCorrection() {}

    // Main correction application method to be implemented by
    // derived classes. Arguments are as follows:
    //
    //
    // Some of the input arguments may be ignored by derived classes.
    //
    void apply(const CaloSamples & cs, const std::vector<int> & capidvec, /*const HcalCoder & coder,*/
                       const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const;

    // Comparison operators. Note that they are not virtual and should
    // not be overriden by derived classes. These operators are very
    // useful for I/O testing.
    inline bool operator==(const PulseShapeFitOOTPileupCorrection& r) const
        {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
    inline bool operator!=(const PulseShapeFitOOTPileupCorrection& r) const
        {return !(*this == r);}

    void setPulseShapeTemplate(const HcalPulseShapes::Shape& ps) {
       spsf_.reset(new FitterFuncs::SinglePulseShapeFunctor(ps));
       dpsf_.reset(new FitterFuncs::DoublePulseShapeFunctor(ps));
    }

protected:
    // Method needed to compare objects for equality.
    bool isEqual(const PulseShapeFitOOTPileupCorrection&) const;

private:
//    HcalPulseShapes theHcalPulseShapes_;

    std::auto_ptr<FitterFuncs::SinglePulseShapeFunctor> spsf_;
    std::auto_ptr<FitterFuncs::DoublePulseShapeFunctor> dpsf_;

    bool useDataPulseShape_;

    int pulseShapeFit(const std::vector<double> & energyVec, const std::vector<double> & pedenVec, const std::vector<double> &chargeVec, const std::vector<double> &pedVec, const double TSTOTen, std::vector<double> &fitParsVec, const std::auto_ptr<FitterFuncs::SinglePulseShapeFunctor>& spsf, const std::auto_ptr<FitterFuncs::DoublePulseShapeFunctor>& dpsf) const;
};

#endif // PulseShapeFitOOTPileupCorrection_h
