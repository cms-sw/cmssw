//-*-c++-*-
//-*-QQ.cpp-*-
//   Written by James Monk and Andrew Pilkington
////////////////////////////////////////////////////////////////////////////////

#include "GeneratorInterface/ExhumeInterface/interface/QQ.h"

////////////////////////////////////////////////////////////////////////////////
Exhume::QQ::QQ(const edm::ParameterSet& pset):
  TwoSpace(pset){
    std::cout<<std::endl<<"   =QQ production selected="<<std::endl<<std::endl;
  SetThetaMin(acos(0.95));
  Partons[0].Colour = 101;
  Partons[0].AntiColour=0;
  Partons[1].Colour=0;
  Partons[1].AntiColour=101;
  //set default to bb production
  SetQuarkType(5);

  Name = "di-quark";

}
////////////////////////////////////////////////////////////////////////////////
double Exhume::QQ::SubProcess(){
  
  double qMassSq=PartonMass*PartonMass;
  double sintheta =1.0/InvSinTheta;
  double etsq = qMassSq*CosTheta*CosTheta + sintheta*sintheta*sHat/4.0;
  double _Sigma = AlphaS(0.5*SqrtsHat)
    *AlphaS(0.5*SqrtsHat)*qMassSq
    *pow((1 - 4*qMassSq*InvsHat),1.5)/(24.0*etsq*etsq);

  
  _Sigma = _Sigma*Gev2fb;
  //val / by 2Pi so that integrate over Phi later

  return(_Sigma); 
}

void Exhume::QQ::LIPS2Amp(){
  double Theta = acos(CosTheta);
  InvSinTheta = 1.0/sin(Theta);

  return;
}

void Exhume::QQ::Amp2LIPS(){
  return;
}

/*
////////////////////////////////////////////////////////////////////////////////
void Exhume::QQ::SetSubParameters(){
  double ThetaRand = double(rand())/RAND_MAX;
  CosTheta = GetValue(ThetaRand);
  Theta = acos(CosTheta);
  SinTheta = sin(Theta);
  InvSinTheta = 1.0/sin(Theta);
}
////////////////////////////////////////////////////////////////////////////////
double Exhume::QQ::SubParameterRange(){
  return(TotalIntegral);
}
////////////////////////////////////////////////////////////////////////////////
void Exhume::QQ::MaximiseSubParameters(){
  WeightInit(CosThetaMin,CosThetaMax);
  Theta = ThetaMin;
  InvSinTheta = 1.0/SinThetaMin;
}
////////////////////////////////////////////////////////////////////////////////
double Exhume::QQ::SubParameterWeight(){
  SubWgt = GetFunc(CosTheta);
  return(SubWgt);
}
////////////////////////////////////////////////////////////////////////////////
void Exhume::QQ::SetPartons(){
  E = 0.5*SqrtsHat;
  Phi = 2*PI*double(rand())/RAND_MAX;
  P = sqrt(E*E-qMassSq);
	Px = P*SinTheta*cos(Phi);
  Py = P*SinTheta*sin(Phi);
  Pz = P*CosTheta;
  
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
/////////////////////////////////////////////////////////////////////////////// 
void Exhume::QQ::SetThetaMin(const double& ThetaMin_){
  ThetaMin = ThetaMin_;
  ThetaMax = PI - ThetaMin_;
  SinThetaMin = sin(ThetaMin);
  CosThetaMin = cos(ThetaMax);
  CosThetaMax = cos(ThetaMin); 
  return;
}
////////////////////////////////////////////////////////////////////////////////
double Exhume::QQ::WeightFunc(const double& CosTheta_){
  InvSinTheta = 1.0/sin(acos(CosTheta_));
  return(SinThetaMin*SinThetaMin*SinThetaMin*SinThetaMin
	 *InvSinTheta*InvSinTheta*InvSinTheta*InvSinTheta);
}

*/
////////////////////////////////////////////////////////////////////////////////
void Exhume::QQ::SetQuarkType(const int& _id){
  Partons[0].id = _id;
  Partons[1].id = -_id;
	switch(_id){
  case 1:
    PartonMass = 0.0;
    break;
  case 2:
    PartonMass = 0.0;
    break;
  case 3:
    PartonMass = StrangeMass;
    break;
  case 4:
    PartonMass = CharmMass;
    break;
  case 5:
    PartonMass = BottomMass;
    break;
  case 6:
    PartonMass = TopMass;
    break;
  default:
    std::cout<<"\tYou have not entered a quark KF code"<<std::endl;
    std::cout<<"\tBottom Production Chosen"<<std::endl;
    PartonMass = BottomMass;
		Partons[0].id = 5;
		Partons[1].id = -5;
    break;
  }
  return;
}
