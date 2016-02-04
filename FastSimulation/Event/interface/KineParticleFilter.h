#ifndef FastSimulation_Event_KineParticleFilter_H
#define FastSimulation_Event_KineParticleFilter_H

//FAMOS Headers
#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"

/**
 * A filter for particles in the user-defined kinematic acceptabce.
 * \author Patrick Janot
 */


#include <set>

namespace edm { 
  class ParameterSet;
}

class KineParticleFilter : public BaseRawParticleFilter {
public:
  KineParticleFilter(const edm::ParameterSet& kine); 
  virtual ~KineParticleFilter(){;};

  void setMainVertex(const XYZTLorentzVector& mv) { mainVertex=mv; }

  const XYZTLorentzVector& vertex() const { return mainVertex; }

private:
  /// the real selection is done here
  virtual bool isOKForMe(const RawParticle* p) const;

  double etaMin, etaMax, phiMin, phiMax, pTMin, pTMax, EMin, EMax;
  double cos2Max, cos2PreshMin, cos2PreshMax;
  XYZTLorentzVector mainVertex;

  std::set<int>   forbiddenPdgCodes;
};

#endif
