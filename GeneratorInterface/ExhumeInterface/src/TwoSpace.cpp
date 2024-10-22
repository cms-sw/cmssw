#include "GeneratorInterface/ExhumeInterface/interface/TwoSpace.h"
#include "CLHEP/Random/RandomEngine.h"

Exhume::TwoSpace::TwoSpace(const edm::ParameterSet& pset) : CrossSection(pset) {
  Partons.resize(2);

  dirty_weighting = true;
  PhiMax = 0.0;
  PartonMass = 0.0;  //default, override in constructor of derived class
  MaximumSubProcessValue = 0.0;
  SetMassAtThetaScan(2000);
  Fudge = 1.2;
}

Exhume::TwoSpace::~TwoSpace() {}

int Exhume::TwoSpace::GetNumberOfSubParameters() { return (2); }

void Exhume::TwoSpace::SetSubParameters() {
  double cos_th = randomEngine->flat();

  CosTheta = GetValue(cos_th);
  Phi = 2 * M_PI * randomEngine->flat();

  //std::cout<<CosTheta<<std::endl;
  LIPS2Amp();
}

void Exhume::TwoSpace::MaximiseSubParameters() {
  if (dirty_weighting == true) {
    //  std::cout<<CosThetaMax<<std::endl;
    WeightInit(CosThetaMin, CosThetaMax);
    dirty_weighting = false;
  }
  CosTheta = MaximumSubProcessCosTheta;
  //  std::cout<<CosTheta<<std::endl;
  Phi = PhiMax;
  LIPS2Amp();
  return;
}

void Exhume::TwoSpace::SetPartons() {
  Amp2LIPS();  //set lips parameters, not needed?
  double _SinTheta = sin(acos(CosTheta));
  double _E = 0.5 * SqrtsHat;
  double _Pmod = sqrt(_E * _E - PartonMass * PartonMass);
  double _Px = _Pmod * _SinTheta * cos(Phi);
  double _Py = _Pmod * _SinTheta * sin(Phi);
  double _Pz = _Pmod * CosTheta;

  Partons[0].p.setPx(_Px);
  Partons[0].p.setPy(_Py);
  Partons[0].p.setPz(_Pz);
  Partons[0].p.setE(_E);

  Partons[1].p.setPx(-_Px);
  Partons[1].p.setPy(-_Py);
  Partons[1].p.setPz(-_Pz);
  Partons[1].p.setE(_E);

  Partons[0].p.boost(CentralVector.boostVector());
  Partons[1].p.boost(CentralVector.boostVector());

  return;
}

double Exhume::TwoSpace::SubParameterWeight() {
  //bit of a misnomer SubParameterWeight tells the Event class the difference between the subprocess Maximum and this specific chance
  Amp2LIPS();  //not needed?
  double _SubWgt;
  _SubWgt = GetFunc(CosTheta);

  // std::cout<<_SubWgt/MaximumSubProcessValue<<std::endl;
  return (Fudge * _SubWgt / (MaximumSubProcessValue));
}

double Exhume::TwoSpace::SubParameterRange() { return (Fudge * 2 * M_PI * TotalIntegral / MaximumSubProcessValue); }
//
//double Exhume::TwoSpace::SubProcess(){
//return(AmplitudeSq*TwoParticleLIPS*Amp2CrossSection);
//define overall factor in font of Msq + DLIPS factor
//}
double Exhume::TwoSpace::WeightFunc(const double& _CosTheta) {
  //arbitrary weighting at low mass
  //std::cout<<_CosTheta<<std::endl;
  CosTheta = _CosTheta;
  //std::cout<<CosTheta<<std::endl;
  Phi = PhiMax;  //arbitrary angle but amp might use it but shouldn'

  //double wgt_hold=0.0;
  //for(int ii=0;ii<100;ii++){
  //double _SqrtsHat = (MassAtThetaScanHigh - MassAtThetaScanLow)*double(ii)/10
  // + MassAtThetaScanLow;
  //double _SqrtsHat=sqrt(4*PartonMass*PartonMass +
  //		MassAtThetaScan*MassAtThetaScan);
  double _SqrtsHat;
  if (PartonMass > 2.0) {
    _SqrtsHat = 2 * PartonMass + 0.001;
  } else {
    _SqrtsHat = sqrt(4 * PartonMass * PartonMass + 10.0);
  }

  SetKinematics(_SqrtsHat, 0.0, 0.0, 0.0, 0.0, 0.0);
  //std::cout<<"here"<<std::endl;
  LIPS2Amp();                  //
                               //std::cout<<"here"<<std::endl;
  double _wgt = SubProcess();  //AmplitudeSq();
  //if(_wgt > wgt_hold){
  //wgt_hold=wgt;
  //}

  if (_wgt > MaximumSubProcessValue) {
    MaximumSubProcessCosTheta = _CosTheta;
    MaximumSubProcessValue = _wgt;
    std::cout << MaximumSubProcessCosTheta << std::endl;
    std::cout << MaximumSubProcessValue << std::endl;
  }
  //std::cout<<_wgt<<"\t"<<CosTheta<<std::endl;
  return (_wgt);
}

void Exhume::TwoSpace::SetThetaMin(const double& ThetaMin_) {
  ThetaMin = ThetaMin_;
  ThetaMax = M_PI - ThetaMin_;
  // SinThetaMin = sin(ThetaMin);
  CosThetaMin = cos(ThetaMax);
  CosThetaMax = cos(ThetaMin);
  MaximumSubProcessCosTheta = CosThetaMax;

  //std::cout<<"called\t"<<CosThetaMin<<std::endl;

  return;
}

//Params outside in event are M,y,t1,t2,phi1,phi2
