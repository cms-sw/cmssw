//-*-c++-*-
//-*-TwoSpace.h-*-
//   Written by James Monk and Andrew Pilkington - 20/08/05

#ifndef TWOSPACE_HH
#define TWOSPACE_HH

#include "GeneratorInterface/ExhumeInterface/interface/CrossSection.h"
#include "GeneratorInterface/ExhumeInterface/interface/Weight.h"

namespace Exhume {

  class TwoSpace : public CrossSection, Weight {
  public:
    TwoSpace(const edm::ParameterSet &);
    ~TwoSpace() override;
    double SubParameterRange() override;
    void MaximiseSubParameters() override;
    void SetSubParameters() override;
    void SetPartons() override;
    void SetMassAtThetaScan(double _M1) {
      MassAtThetaScan = _M1;
      return;
    }
    double SubProcess() override = 0;
    double SubParameterWeight() override;
    void SetThetaMin(const double &);
    int GetNumberOfSubParameters();
    //allows user to define an amplitude (Msq) and use all our pre defined funcs
    //virtual double AmplitudeSq()=0;
    virtual void Amp2LIPS() = 0;
    virtual void LIPS2Amp() = 0;
    double WeightFunc(const double &) override;

  protected:
    double CosTheta, Phi;  //lips parameters
    double ThetaMin, ThetaMax;
    double MaximumSubProcessValue, MaximumSubProcessCosTheta;
    double PhiMax, CosThetaMax, CosThetaMin;
    double PartonMass;
    bool dirty_weighting;
    double MassAtThetaScan;
    double Fudge;
  };
}  // namespace Exhume

#endif
