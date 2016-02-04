#include "FastSimulation/Particle/interface/RawStableParticleFilter.h"

bool RawStableParticleFilter::isOKForMe(const RawParticle* p) const
{
  return (p->status() == 1) ;
}
