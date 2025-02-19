#ifndef RAWSTABLEPARTTICLEFILTER_H
#define RAWSTABLEPARTTICLEFILTER_H

#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

/**
 * A filter for stable particles.
 * Stable means particle.status() is 1.
 * \author Stephan Wynhoff
 */
class RawStableParticleFilter : public BaseRawParticleFilter {
public:
  RawStableParticleFilter(){;}; 
  virtual ~RawStableParticleFilter(){;};
private:
  virtual bool isOKForMe(const RawParticle* p) const;
};

#endif
