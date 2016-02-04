//-*-c++-*-
//-*-DiPhoton.cpp-*-ahamil@localhost
//   Written by James Monk and Andrew Pilkington
////////////////////////////////////////////////////////////////////////////////

#include "GeneratorInterface/ExhumeInterface/interface/DiPhoton.h"

////////////////////////////////////////////////////////////////////////////////
Exhume::DiPhoton::DiPhoton(const edm::ParameterSet& pset):
  TwoSpace(pset){
    std::cout<<std::endl<<"   =DiPhoton production selected="
	     <<std::endl<<std::endl;
  Partons[0].id = 22;
  Partons[1].id = 22;
  Partons[0].Colour = 0;
  Partons[0].AntiColour=0;
  Partons[1].Colour=0;
  Partons[1].AntiColour=0;
  Inv64PI2=double(1.0)/(64.0*PI*PI);
  PI2 = PI * PI;

  Nup=2;
  Ndown=2;
  MatFact=4.0*(2*double(16.0)/81 + 2*double(1.0)/81)*Inv64PI2;
  
  Name = "di-photon";
  SetThetaMin(acos(0.95));
}
////////////////////////////////////////////////////////////////////////////////
double Exhume::DiPhoton::MatrixElement(){
    
  //M++++ + M--++ can interfere because of same final states
  double _Sigma = 
    -0.5*(0.5*(1+CosTheta*CosTheta))*
    (log((1.0-CosTheta)/(1.0+CosTheta))*
     log((1.0-CosTheta)/(1.0+CosTheta)) +PI2)
    - CosTheta*log((1.0-CosTheta)/(1.0+CosTheta)); 
  
  //Total M^2 = 2* above (helicity opposites) plus 2*(M+++- + M--+-)
  // note M+++- = M++-+ = 1 (same for helicity opposites
  return(2*_Sigma*_Sigma + 8.0);
}
////////////////////////////////////////////////////////////////////////////////

double Exhume::DiPhoton::SubProcess(){
  double m_elem_sq = MatrixElement();
  double _Sigma = MatFact*(m_elem_sq)/sHat;
  _Sigma = _Sigma*AlphaS(SqrtsHat/2)*AlphaS(SqrtsHat/2)*AlphaEw*AlphaEw;
  return(Gev2fb*_Sigma);

}

void Exhume::DiPhoton::LIPS2Amp(){
  return;

}

void Exhume::DiPhoton::Amp2LIPS(){
  return;
}
