/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef TrackParticle_h
#define TrackParticle_h

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "TString.h"

class TrackParticle : public Particle {
 public:
  enum {kappa=0,lambda,phi,dxy,dz,NHelixPar};// 5 track helix Parameters

  TrackParticle(TMatrixT<double> par_, TMatrixTSym<double> cov_, int pdgid_, double mass_,double charge_, double b_);
  virtual ~TrackParticle(){};

  static TString Name(int i);
  virtual int NParameters(){return NHelixPar;}
  virtual double Mass(){return mass;}
  
 private:
  double mass;
};
#endif


