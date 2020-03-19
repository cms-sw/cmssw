//-*-c++-*-
//-*-GG.cpp-*-
//   Written by James Monk and Andrew Pilkington
//////////////////////////////////////////////////////////////////////////////

#include "GeneratorInterface/ExhumeInterface/interface/GG.h"

//////////////////////////////////////////////////////////////////////////////
Exhume::GG::GG(const edm::ParameterSet& pset) : TwoSpace(pset) {
  std::cout << std::endl << "   =Glu Glu production selected=" << std::endl;
  SetThetaMin(acos(0.95));
  Partons[0].id = 21;
  Partons[1].id = 21;
  Partons[0].Colour = 101;
  Partons[0].AntiColour = 102;
  Partons[1].Colour = 102;
  Partons[1].AntiColour = 101;

  EighteenPI = 18.0 * M_PI;

  Name = "di-gluon";
}
//////////////////////////////////////////////////////////////////////////////
double Exhume::GG::SubProcess() {
  //error ? 0.5 factor from integration over t -> cos theta?
  double AlphaS_ = AlphaS(0.5 * SqrtsHat);
  double InvSinTheta2 = InvSinTheta * InvSinTheta;
  return (EighteenPI * AlphaS_ * AlphaS_ * InvsHat * InvSinTheta2 * InvSinTheta2 * Gev2fb / (2 * M_PI));
}
void Exhume::GG::LIPS2Amp() {
  double Theta = acos(CosTheta);
  //SinTheta = sin(Theta);
  InvSinTheta = 1.0 / sin(Theta);

  return;
}

void Exhume::GG::Amp2LIPS() { return; }
/*
//////////////////////////////////////////////////////////////////////////////
void Exhume::GG::SetSubParameters(){
  double ThetaRand = double(rand())/RAND_MAX;
  CosTheta = GetValue(ThetaRand);
  Theta = acos(CosTheta);
  SinTheta = sin(Theta);
  InvSinTheta = 1.0/sin(Theta);
}
//////////////////////////////////////////////////////////////////////////////
double Exhume::GG::SubParameterRange(){
  return(TotalIntegral);
}
//////////////////////////////////////////////////////////////////////////////
void Exhume::GG::MaximiseSubParameters(){
  WeightInit(CosThetaMin,CosThetaMax);
  Theta = ThetaMin;
  InvSinTheta = 1.0/SinThetaMin;
}
//////////////////////////////////////////////////////////////////////////////
double Exhume::GG::SubParameterWeight(){
  SubWgt = GetFunc(CosTheta);
  return(SubWgt);
}
//////////////////////////////////////////////////////////////////////////////
void Exhume::GG::SetPartons(){
  E = 0.5*SqrtsHat;
  Phi = 2*M_PI*double(rand())/RAND_MAX;
  Px = E*SinTheta*cos(Phi);
  Py = E*SinTheta*sin(Phi);
  Pz = E*CosTheta;
  
  Partons[0].p.setPx(Px);
  Partons[0].p.setPy(Py);
  Partons[0].p.setPz(Pz);
  Partons[0].p.setE(E);

  Partons[1].p.setPx(-Px);
  Partons[1].p.setPy(-Py);
  Partons[1].p.setPz(-Pz);
  Partons[1].p.setE(E);

  Partons[0].p.boost(CentralVector.boostVector());
  Partons[1].p.boost(CentralVector.boostVector());
  
  return;

}

///////////////////////////////////////////////////////////////////////////// 
void Exhume::GG::SetThetaMin(const double& ThetaMin_){
  ThetaMin = ThetaMin_;
  ThetaMax = M_PI - ThetaMin_;
  // SinThetaMin = sin(ThetaMin);
  CosThetaMin = cos(ThetaMax);
  CosThetaMax = cos(ThetaMin); 
  return;
}


//////////////////////////////////////////////////////////////////////////////
double Exhume::GG::WeightFunc(const double& CosTheta_){
  InvSinTheta = 1.0/sin(acos(CosTheta_));
  return(SinThetaMin*SinThetaMin*SinThetaMin*SinThetaMin
	 *InvSinTheta*InvSinTheta*InvSinTheta*InvSinTheta);
}
//////////////////////////////////////////////////////////////////////////////
*/
