#ifndef FastSimulation_Event_KineParticleFilter_H
#define FastSimulation_Event_KineParticleFilter_H

//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//FAMOS Headers
#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"

/**
 * A filter for particles in the user-defined kinematic acceptabce.
 * \author Patrick Janot
 */

class KineParticleFilter : public BaseRawParticleFilter {
public:
  KineParticleFilter(const edm::ParameterSet& kine); 
  virtual ~KineParticleFilter(){;};

  void setMainVertex(const HepLorentzVector& mv) { mainVertex=mv; }

  const HepLorentzVector& vertex() { return mainVertex; }

private:
  /// the real selection is done here
  virtual bool isOKForMe(const RawParticle* p) const;

  double etaMin, etaMax, phiMin, phiMax, pTMin, pTMax, EMin, EMax;
  double cos2Max, cos2PreshMin, cos2PreshMax;
  HepLorentzVector mainVertex;

};

#endif
