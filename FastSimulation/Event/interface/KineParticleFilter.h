#ifndef KINEPARTTICLEFILTER_H
#define KINEEPARTTICLEFILTER_H

#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"

/**
 * A filter for particles in the user-defined kinematic acceptabce.
 * \author Patrick Janot
 */

class KineParticleFilter : public BaseRawParticleFilter {
public:
  KineParticleFilter(); 
  virtual ~KineParticleFilter(){;};

  void setMainVertex(const HepLorentzVector& mv) { mainVertex=mv; }

  const HepLorentzVector& vertex() { return mainVertex; }

private:
  /// the real selection is done here
  virtual bool isOKForMe(const RawParticle* p) const;

  double etaMin, etaMax, phiMin, phiMax, pTMin, pTMax, EMin, EMax;
  HepLorentzVector mainVertex;

};

#endif
