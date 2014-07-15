#ifndef PythiaDecays_h
#define PythiaDecays_h

#include <memory>
#include <vector>

// Needed for Pythia8
#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h> 
#include <Pythia8/Pythia.h>
#include <Pythia8/Pythia8ToHepMC.h>
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

class ParticlePropagator;
class Pythia6jets;
class RawParticle;

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {
  class P8RndmEngine;
}

typedef std::vector<RawParticle> DaughterParticleList;
typedef DaughterParticleList::const_iterator DaughterParticleIterator; 

namespace gen { class Pythia6Service; } // remove?
//using namespace gen;

class PythiaDecays 
{
 public:
  PythiaDecays(std::string program);
  ~PythiaDecays();

  const DaughterParticleList&
    particleDaughtersPy6(ParticlePropagator& particle, CLHEP::HepRandomEngine*);
  const DaughterParticleList&
    particleDaughtersPy8(ParticlePropagator& particle, CLHEP::HepRandomEngine*);

 private:
  DaughterParticleList theList;
  std::string program_;
  // for Pythia6:
  gen::Pythia6Service *pyservice;
  Pythia6jets* pyjets;
  // for Pythia8:
  std::auto_ptr<Pythia8::Pythia>   decayer; 
  std::unique_ptr<gen::P8RndmEngine> p8RndmEngine;
};
#endif
