#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"

/*
bool BaseRawParticleFilter::accept(const RawParticle& p) const
{
  return this->accept(&p);
}
*/

bool BaseRawParticleFilter::accept(const RawParticle & p) const
{
  return this->isOKForMe(&p) ;
}

/*
void BaseRawParticleFilter::addFilter(BaseRawParticleFilter* f)
{
  myFilter.push_back(f);
}
*/
