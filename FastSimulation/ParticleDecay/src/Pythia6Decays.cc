// HepMC Headers
#include "HepMC/PythiaWrapper6_2.h"

// FAMOS Headers
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6Decays.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6jets.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6Random.h"

#define PYTHIA6PYDECY pythia6pydecy_

extern "C" {
  void PYTHIA6PYDECY(int *ip);
}

Pythia6Decays::Pythia6Decays(int seed,double comE)
{
  // Create a new Pythia6jets
  pyjets = new Pythia6jets();
  // Create a new Pythia6Random steering
  pyrand = new Pythia6Random(seed);
  // Initialize PYTHIA decay tables...
  call_pyinit( "CMS", "p", "p", comE );

}

Pythia6Decays::~Pythia6Decays() {
  delete pyjets;
  delete pyrand;
}

const void 
Pythia6Decays::getRandom() {
  pyrand->save(0);
  pyrand->get(1);
}

const void 
Pythia6Decays::saveRandom() {
  pyrand->save(1);
  pyrand->get(0);
}

const DaughterParticleList&
Pythia6Decays::particleDaughters(ParticlePropagator& particle)
{
  //  Pythia6jets pyjets;
  int ip;

  pyjets->k(1,1) = 1;
  pyjets->k(1,2) = particle.pid();
  pyjets->p(1,1) = particle.Px();
  pyjets->p(1,2) = particle.Py();
  pyjets->p(1,3) = particle.Pz();
  pyjets->p(1,4) = std::max(particle.mass(),particle.e());
  pyjets->p(1,5) = particle.mass();
  pyjets->v(1,1) = particle.X();
  pyjets->v(1,2) = particle.Y();
  pyjets->v(1,3) = particle.Z();
  pyjets->v(1,4) = particle.T();
  pyjets->n() = 1;
  
  ip = 1;
  PYTHIA6PYDECY(&ip);

  // Fill the list of daughters
  theList.clear();
  if ( pyjets->n()==1 ) return theList; 

  theList.resize(pyjets->n()-1,RawParticle());

  for (int i=2;i<=pyjets->n();++i) {
    
    theList[i-2].SetXYZT(pyjets->p(i,1),pyjets->p(i,2),
			 pyjets->p(i,3),pyjets->p(i,4)); 
    theList[i-2].setVertex(pyjets->v(i,1),pyjets->v(i,2),
			   pyjets->v(i,3),pyjets->v(i,4));
    theList[i-2].setID(pyjets->k(i,2));
    theList[i-2].setMass(pyjets->p(i,5));

  }

  return theList;
  
}
