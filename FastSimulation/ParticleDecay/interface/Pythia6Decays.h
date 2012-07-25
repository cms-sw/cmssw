#ifndef Pythia6Decays_h
#define Pythia6Decays_h

#include <vector>

class ParticlePropagator;
class Pythia6jets;
class RawParticle;

typedef std::vector<RawParticle> DaughterParticleList;
typedef DaughterParticleList::const_iterator DaughterParticleIterator; 

namespace gen { class Pythia6Service; }

class Pythia6Decays 
{
 public:
  Pythia6Decays();
  ~Pythia6Decays();

  const DaughterParticleList&
    particleDaughters(ParticlePropagator& particle);

 private:

  gen::Pythia6Service *pyservice;
  Pythia6jets* pyjets;
  DaughterParticleList theList;

};
#endif
