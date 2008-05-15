#ifndef Pythia6Decays_h
#define Pythia6Decays_h

#include <vector>

class ParticlePropagator;
class Pythia6jets;
class Pythia6Random;
class RawParticle;

typedef std::vector<RawParticle> DaughterParticleList;
typedef DaughterParticleList::const_iterator DaughterParticleIterator; 

class Pythia6Decays 
{
 public:
  Pythia6Decays(int seed,double comE=14000.);
  ~Pythia6Decays();

  const DaughterParticleList&
    particleDaughters(ParticlePropagator& particle);

  const void getRandom();
  const void saveRandom();

 private:

  Pythia6jets* pyjets;
  Pythia6Random* pyrand;
  DaughterParticleList theList;

};
#endif
