#ifndef PythiaDecays_h
#define PythiaDecays_h

#include <memory>
#include <vector>

// TrajectoryManager does not compile when forward declaring P8RndmEngine
#include "GeneratorInterface/Pythia8Interface/interface/P8RndmEngine.h"

class ParticlePropagator;
class RawParticle;

namespace gen {
  class P8RndmEngine;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace Pythia8 {
  class Pythia;
}

typedef std::vector<RawParticle> DaughterParticleList;
typedef DaughterParticleList::const_iterator DaughterParticleIterator; 

class PythiaDecays 
{
 public:

  PythiaDecays();
  ~PythiaDecays();
  const DaughterParticleList & particleDaughters(ParticlePropagator& particle, CLHEP::HepRandomEngine*);

 private:

  DaughterParticleList theList;
  std::unique_ptr<Pythia8::Pythia>   decayer; 
  std::unique_ptr<gen::P8RndmEngine> p8RndmEngine;
};
#endif
