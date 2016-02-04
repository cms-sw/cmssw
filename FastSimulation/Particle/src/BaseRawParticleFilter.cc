#include "FastSimulation/Particle/interface/BaseRawParticleFilter.h"

bool BaseRawParticleFilter::accept(const RawParticle& p) const
{
  return this->accept(&p);
}

bool BaseRawParticleFilter::accept(const RawParticle* p) const
{
  //  cout << "test a particle pointer" << endl;
  bool acceptThis = false;

  acceptThis = this->isOKForMe(p) ;
  
  std::vector<BaseRawParticleFilter*>::const_iterator myFilterItr;
  myFilterItr = myFilter.begin();

  while ( acceptThis && 
	  ( myFilterItr != myFilter.end() ) ) {
    acceptThis = acceptThis && (*myFilterItr)->accept(p);
    myFilterItr++;
  }
  return acceptThis;
}

void BaseRawParticleFilter::addFilter(BaseRawParticleFilter* f)
{
  myFilter.push_back(f);
}
