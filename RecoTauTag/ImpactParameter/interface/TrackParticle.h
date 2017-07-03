#ifndef RecoTauTag_ImpactParameter_TrackParticle_h
#define RecoTauTag_ImpactParameter_TrackParticle_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "TString.h"

namespace tauImpactParameter {

class TrackParticle : public Particle {
 public:
  enum {kappa=0,lambda,phi,dxy,dz,NHelixPar};// 5 track helix Parameters

  TrackParticle(const TVectorT<double>& par, const TMatrixTSym<double>& cov, int pdgid, double mass,double charge, double b);
  ~TrackParticle() override{};

  static TString name(int i);
  int nParameters() const override { return NHelixPar; }
  double mass() const override { return mass_; }
  
 private:
  double mass_;
};

}
#endif


