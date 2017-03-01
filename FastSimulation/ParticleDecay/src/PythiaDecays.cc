#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/ParticleDecay/interface/PythiaDecays.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

#include <Pythia8/Pythia.h>
#include "Pythia8Plugins/HepMC2.h"

PythiaDecays::PythiaDecays()
{
    // inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
    decayer.reset(new Pythia8::Pythia);
    p8RndmEngine.reset(new gen::P8RndmEngine);
    decayer->setRndmEnginePtr(p8RndmEngine.get());
    decayer->settings.flag("ProcessLevel:all",false);
    decayer->settings.flag("PartonLevel:FSRinResonances",false);
    decayer->settings.flag("ProcessLevel:resonanceDecays",false);
    decayer->init();

    // forbid all decays
    // (decays are allowed selectively in the particleDaughters function)
    Pythia8::ParticleData & pdt = decayer->particleData;
    int pid = 1;
    while(pdt.nextId(pid) > pid){
      pid = pdt.nextId(pid);
      pdt.mayDecay(pid,false);
    }
}

const DaughterParticleList&
PythiaDecays::particleDaughters(ParticlePropagator& particle, CLHEP::HepRandomEngine* engine)
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
