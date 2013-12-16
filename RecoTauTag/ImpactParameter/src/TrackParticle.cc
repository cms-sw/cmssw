/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"

TrackParticle::TrackParticle(TMatrixT<double> par_, TMatrixTSym<double> cov_, int pdgid_,double mass_, double charge_, double b_):
  Particle(par_,cov_,pdgid_,charge_,b_),
  mass(mass_)
{

}

TString TrackParticle::Name(int i){
  if(i==kappa)  return "kappa";
  if(i==lambda) return "lambda";
  if(i==phi)    return "phi";
  if(i==dz)     return "dz";
  if(i==dxy)    return "dxy";
  return "invalid";
}
