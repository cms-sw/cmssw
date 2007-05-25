#ifndef Pythia6Decays_h
#define Pythia6Decays_h

#include <list>

class ParticlePropagator;
class Pythia6jets;
class RawParticle;

typedef std::list<const RawParticle *> DaughterParticleList;
typedef DaughterParticleList::const_iterator DaughterParticleIterator; 

class Pythia6Decays 
{
 public:
  Pythia6Decays();
  ~Pythia6Decays();

  const DaughterParticleList&
    particleDaughters(ParticlePropagator& particle);

 private:

  Pythia6jets* pyjets;
  DaughterParticleList theList;


};
#endif
