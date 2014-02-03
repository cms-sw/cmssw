#ifndef PythiaDecays_h
#define PythiaDecays_h

#include <vector>

// Needed for Pythia8
#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h> 
#include <Pythia.h>
#include <HepMCInterface.h>
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

class ParticlePropagator;
class Pythia6jets;
class RawParticle;

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
    particleDaughtersPy6(ParticlePropagator& particle);
  const DaughterParticleList&
    particleDaughtersPy8(ParticlePropagator& particle);

 private:
  DaughterParticleList theList;
  std::string program_;
  // for Pythia6:
  gen::Pythia6Service *pyservice;
  Pythia6jets* pyjets;
  // for Pythia8:
  std::auto_ptr<Pythia8::Pythia>   pythia;
  std::auto_ptr<Pythia8::Pythia>   decayer; 

};
#endif
