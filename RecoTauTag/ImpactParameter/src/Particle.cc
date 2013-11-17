#include "RecoTauTag/ImpactParameter/interface/Particle.h"

Particle::Particle(TMatrixT<double> par_, TMatrixTSym<double> cov_, int pdgid_, double charge_, double b_):
  par(par_),
  cov(cov_),
  b(b_),
  charge(charge_),
  pdgid(pdgid_)
{

}
