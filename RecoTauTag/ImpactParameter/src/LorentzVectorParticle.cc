#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"


LorentzVectorParticle::LorentzVectorParticle():
  Particle(TMatrixT<double>(NLorentzandVertexPar,1),TMatrixTSym<double>(NLorentzandVertexPar),0,0,0)
{
}


LorentzVectorParticle::LorentzVectorParticle(TMatrixT<double> par_,TMatrixTSym<double> cov_,int pdgid_,double charge_,double b_):
  Particle(par_,cov_,pdgid_,charge_,b_)
{
}


TString LorentzVectorParticle::Name(int i){
  if(i==px)  return "px";
  if(i==py)  return "py";
  if(i==pz)  return "pz";
  if(i==m)   return "m";
  if(i==vx)  return "vx";
  if(i==vy)  return "vy";
  if(i==vz)  return "vz";
  return "invalid";
}
