// HepMC Headers
#include "HepMC/PythiaWrapper6_4.h"

// Pythia6 framework integration service Headers
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

// FAMOS Headers
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticleDecay/interface/PythiaDecays.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6jets.h"

#include "GeneratorInterface/Pythia8Interface/interface/P8RndmEngine.h"

#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

// Needed for Pythia6 
#define PYTHIA6PYDECY pythia6pydecy_

extern "C" {
  void PYTHIA6PYDECY(int *ip);
}

PythiaDecays::PythiaDecays(std::string program)
{
  program_=program;
  if (program_ == "pythia6") {
    //// Pythia6:
    pyjets = new Pythia6jets();
    pyservice = new gen::Pythia6Service();
    // The PYTHIA decay tables will be initialized later 
  } else if (program_ == "pythia8") {

    //// Pythia8:

    // inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
    decayer.reset(new Pythia8::Pythia);
    p8RndmEngine.reset(new gen::P8RndmEngine);
    decayer->setRndmEnginePtr(p8RndmEngine.get());
    decayer->readString("ProcessLevel:all = off");
    decayer->readString("PartonLevel:FSRinResonances = off"); //?
    decayer->readString("ProcessLevel:resonanceDecays = off"); //?
    decayer->init();

    // forbid all decays    
    Pythia8::ParticleData & pdt = decayer->particleData;
    int pid = 1;
    while(pdt.nextId(pid) > pid){
      pid = pdt.nextId(pid);
      pdt.mayDecay(pid,false);
    }

  } else {
    std::cout << "WARNING: you are requesting an option which is not available in PythiaDecays::PythiaDecays " << std::endl;
  }

}

PythiaDecays::~PythiaDecays() {
  if (program_ == "pythia6") {
    delete pyjets;
    delete pyservice;
  }
}

const DaughterParticleList&
PythiaDecays::particleDaughtersPy8(ParticlePropagator& particle, CLHEP::HepRandomEngine* engine)
{
  edm::RandomEngineSentry<gen::P8RndmEngine> sentry(p8RndmEngine.get(), engine);

  theList.clear();

  // inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
  int pid = particle.pid();
  decayer->event.reset();
  Pythia8::Particle py8part( pid , 93, 0, 0, 0, 0, 0, 0,
		     particle.momentum().x(), // note: momentum().x() and Px() are the same
		     particle.momentum().y(),
		     particle.momentum().z(),
		     particle.momentum().t(),
		     particle.mass() );
  py8part.vProd( particle.X(), particle.Y(), 
		 particle.Z(), particle.T() );
  decayer->event.append( py8part );

  int nentries_before = decayer->event.size();
  decayer->particleData.mayDecay(pid,true);   // switch on the decay of this and only this particle (avoid double decays)
  decayer->next();                           // do the decay
  decayer->particleData.mayDecay(pid,false);  // switch it off again
  int nentries_after = decayer->event.size();
  if ( nentries_after <= nentries_before ) return theList;

  theList.resize(nentries_after - nentries_before,RawParticle());


  for ( int ipart=nentries_before; ipart<nentries_after; ipart++ )
    {
      Pythia8::Particle& py8daughter = decayer->event[ipart];
      theList[ipart-nentries_before].SetXYZT( py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e() );
      theList[ipart-nentries_before].setVertex( py8daughter.xProd(),
					   py8daughter.yProd(),
					   py8daughter.zProd(),
					   py8daughter.tProd() );
      theList[ipart-nentries_before].setID( py8daughter.id() );
      theList[ipart-nentries_before].setMass( py8daughter.m() );
    }

  return theList;
}

const DaughterParticleList&
PythiaDecays::particleDaughtersPy6(ParticlePropagator& particle, CLHEP::HepRandomEngine* engine)
{
  edm::RandomEngineSentry<gen::Pythia6Service> sentry(pyservice, engine);

  gen::Pythia6Service::InstanceWrapper guard(pyservice); // grab Py6 context

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
