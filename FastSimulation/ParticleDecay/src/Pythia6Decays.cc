// HepMC Headers
#include "HepMC/PythiaWrapper6_2.h"

// FAMOS Headers
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6Decays.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6jets.h"

#define PYTHIA6PYDECY pythia6pydecy_

extern "C" {
  void PYTHIA6PYDECY(int *ip);
}

Pythia6Decays::Pythia6Decays()
{
  // Initialize PYTHIA decay tables...
  call_pyinit( "CMS", "p", "p", 14000. );
  // Create a new Pythia6jets
  pyjets = new Pythia6jets();
}

Pythia6Decays::~Pythia6Decays() {
  delete pyjets;
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

  // Some cleaning : delete the daughter pointers
  // delete the pointeurs
  for( DaughterParticleIterator deleteiter = theList.begin(); 
                                deleteiter!= theList.end(); 
                              ++deleteiter ) {
    delete (*deleteiter);
  }
  // Clear the List of daughters
  theList.clear();

  for (int i=2;i<=pyjets->n();++i) {
    
    XYZTLorentzVector fourvector, fourvector1;
    
    fourvector.SetXYZT(pyjets->p(i,1),pyjets->p(i,2),pyjets->p(i,3),pyjets->p(i,4));
    fourvector1.SetXYZT(pyjets->v(i,1),pyjets->v(i,2),pyjets->v(i,3),pyjets->v(i,4));
    RawParticle *aNewParticle = new RawParticle(fourvector,fourvector1);
    aNewParticle->setID(pyjets->k(i,2));
    aNewParticle->setMass(pyjets->p(i,5));
    theList.push_back(aNewParticle);
  }

  return theList;
  
}
