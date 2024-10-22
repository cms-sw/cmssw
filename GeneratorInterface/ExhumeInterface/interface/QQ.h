//-*-c++-*-
//-*-QQ.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#ifndef QQ_HH
#define QQ_HH

#include "GeneratorInterface/ExhumeInterface/interface/TwoSpace.h"

namespace Exhume {
  class QQ : public TwoSpace {
  public:
    QQ(const edm::ParameterSet&);

    //declare inherited functions
    double SubProcess() override;
    void LIPS2Amp() override;
    void Amp2LIPS() override;
    void SetQuarkType(const int&);

  private:
    double InvSinTheta;

    /*
    void SetPartons();
    void SetSubParameters();
    double SubParameterWeight();
    void MaximiseSubParameters();
    double SubParameterRange();
      
    //declare sub-process specific functions
    void SetThetaMin(const double&);
		void SetQuarkType(const int&);
  private:
    //internal functions
    double WeightFunc(const double&);
    //internal Parameters
    double qMass,qMassSq;
    double CosTheta,CosThetaMin,CosThetaMax;
    double ThetaMin,ThetaMax, SinThetaMin;
    double Theta,SinTheta,InvSinTheta; 
    double E,P,Px,Py,Pz,Phi;
    double SubWgt;
    */
  };
}  // namespace Exhume
#endif
