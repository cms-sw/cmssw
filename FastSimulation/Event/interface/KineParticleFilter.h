#ifndef FastSimulation_Event_KineParticleFilter_H
#define FastSimulation_Event_KineParticleFilter_H

#include "DataFormats/Math/interface/LorentzVector.h"

class RawParticle;
namespace edm {
  class ParameterSet;
}

class KineParticleFilter {
public:
  KineParticleFilter(const edm::ParameterSet& kine);

  ~KineParticleFilter() { ; }

  bool acceptParticle(const RawParticle& p) const;

  bool acceptVertex(const math::XYZTLorentzVector& p) const;

private:
  // see constructor for comments
  double chargedPtMin2, EMin, protonEMin;
  double cos2ThetaMax;
  double vertexRMax2, vertexZMax;
};

#endif
