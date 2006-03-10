#ifndef KINEPARTTICLEFILTER_H
#define KINEEPARTTICLEFILTER_H

#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

/**
 * A filter for particles in the user-defined kinematic acceptabce.
 * \author Patrick Janot
 */

class KineParticleFilter : public BaseRawParticleFilter {
public:
  KineParticleFilter(); 
  virtual ~KineParticleFilter(){;};

  /// accept a particle from its type, momentum and origin vertex
  virtual bool accept(int pid, const HepLorentzVector& p, 
		               const HepLorentzVector& v) const;

  /// accept a vertex from its position and the mother particle type 
  virtual bool accept(int pid, const HepLorentzVector& v) const; 

  /// accept a particle from the particle type 
  virtual bool accept(int pid) const; 

  void setMainVertex(const HepLorentzVector& mv) { mainVertex=mv; }

  const HepLorentzVector& vertex() { return mainVertex; }

private:
  /// the real selection is done here
  virtual bool isOKForMe(const RawParticle* p) const;

  double etaMin, etaMax, phiMin, phiMax, pTMin, pTMax, EMin, EMax;
  HepLorentzVector mainVertex;

};

#endif
