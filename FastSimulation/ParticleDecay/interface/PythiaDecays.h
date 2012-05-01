#ifndef PythiaDecays_h
#define PythiaDecays_h

#include <vector>

class ParticlePropagator;
class Pythia6jets;
class RawParticle;

typedef std::vector<RawParticle> DaughterParticleList;
typedef DaughterParticleList::const_iterator DaughterParticleIterator; 

namespace gen { class Pythia6Service; }

class PythiaDecays 
{
 public:
  PythiaDecays();
  ~PythiaDecays();

  const DaughterParticleList&
    particleDaughtersPy6(ParticlePropagator& particle);
  const DaughterParticleList&
    particleDaughtersPy8(ParticlePropagator& particle);

 private:

  gen::Pythia6Service *pyservice;
  Pythia6jets* pyjets;
  DaughterParticleList theList;

};
#endif
