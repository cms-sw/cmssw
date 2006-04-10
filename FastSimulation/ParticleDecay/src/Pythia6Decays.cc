// CLHEP Headers
#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/HepMC/include/PythiaWrapper6_2.h"
// FAMOS Headers
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6Decays.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6jets.h"

#include <list>
#include <algorithm>

#define PYTHIA6PYDECY pythia6pydecy_

extern "C" {
  void PYTHIA6PYDECY(int *ip);
}

using namespace std;


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
  pyjets->p(1,1) = particle.px();
  pyjets->p(1,2) = particle.py();
  pyjets->p(1,3) = particle.pz();
  pyjets->p(1,4) = std::max(particle.PDGmass(),particle.e());
  pyjets->p(1,5) = particle.PDGmass();
  pyjets->v(1,1) = particle.x();
  pyjets->v(1,2) = particle.y();
  pyjets->v(1,3) = particle.z();
  pyjets->v(1,4) = particle.t();
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
    
    HepLorentzVector fourvector, fourvector1;
    
    fourvector.setPx(pyjets->p(i,1));
    fourvector.setPy(pyjets->p(i,2));
    fourvector.setPz(pyjets->p(i,3));
    fourvector.setE(pyjets->p(i,4));
    fourvector1.setX(pyjets->v(i,1));
    fourvector1.setY(pyjets->v(i,2));
    fourvector1.setZ(pyjets->v(i,3));
    fourvector1.setT(pyjets->v(i,4));
    RawParticle *aNewParticle = new RawParticle(fourvector,fourvector1);
    aNewParticle->setID(pyjets->k(i,2));
    aNewParticle->setMass(pyjets->p(i,5));
    theList.push_back(aNewParticle);
  }

  return theList;
  
}
