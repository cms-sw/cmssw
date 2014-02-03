/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"

using namespace tauImpactParameter;

TrackParticle::TrackParticle(const TVectorT<double>& par, const TMatrixTSym<double>& cov, int pdgid, double mass, double charge, double b)
  : Particle(par,cov,pdgid,charge,b),
    mass_(mass)
{}

TString TrackParticle::name(int i){
  if ( i == kappa  ) return "kappa";
  if ( i == lambda ) return "lambda";
  if ( i == phi    ) return "phi";
  if ( i == dz     ) return "dz";
  if ( i == dxy    ) return "dxy";
  return "invalid";
}
