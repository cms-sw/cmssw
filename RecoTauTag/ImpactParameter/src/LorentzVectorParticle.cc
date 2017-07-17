#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"

using namespace tauImpactParameter;

LorentzVectorParticle::LorentzVectorParticle()
  : Particle(TVectorT<double>(NLorentzandVertexPar),TMatrixTSym<double>(NLorentzandVertexPar),0,0,0)
{}

LorentzVectorParticle::LorentzVectorParticle(const TVectorT<double>& par, const TMatrixTSym<double>& cov, int pdgid ,double charge,double b)
  : Particle(par, cov, pdgid, charge,b)
{}

TString LorentzVectorParticle::name(int i)
{
  if (i == px) return "px";
  if (i == py) return "py";
  if (i == pz) return "pz";
  if (i == m ) return "m";
  if (i == vx) return "vx";
  if (i == vy) return "vy";
  if (i == vz) return "vz";
  return "invalid";
}
