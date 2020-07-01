#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/Particle/interface/makeParticle.h"
#include "FastSimulation/ParticleDecay/interface/PythiaDecays.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

#include <Pythia8/Pythia.h>


#include <memory>

#include "Pythia8Plugins/HepMC2.h"

PythiaDecays::PythiaDecays() {
  // inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
  decayer = std::make_unique<Pythia8::Pythia>();
  p8RndmEngine = std::make_unique<gen::P8RndmEngine>();
  decayer->setRndmEnginePtr(p8RndmEngine.get());
  decayer->settings.flag("ProcessLevel:all", false);
  decayer->settings.flag("PartonLevel:FSRinResonances", false);
  decayer->settings.flag("ProcessLevel:resonanceDecays", false);
  decayer->init();

  // forbid all decays
  // (decays are allowed selectively in the particleDaughters function)
  Pythia8::ParticleData& pdt = decayer->particleData;
  int pid = 1;
  while (pdt.nextId(pid) > pid) {
    pid = pdt.nextId(pid);
    pdt.mayDecay(pid, false);
  }
}

PythiaDecays::~PythiaDecays() {}

const DaughterParticleList& PythiaDecays::particleDaughters(ParticlePropagator& particle,
                                                            CLHEP::HepRandomEngine* engine) {
  edm::RandomEngineSentry<gen::P8RndmEngine> sentry(p8RndmEngine.get(), engine);

  theList.clear();

  // inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
  int pid = particle.particle().pid();
  decayer->event.reset();
  Pythia8::Particle py8part(pid,
                            93,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            particle.particle().momentum().x(),  // note: momentum().x() and Px() are the same
                            particle.particle().momentum().y(),
                            particle.particle().momentum().z(),
                            particle.particle().momentum().t(),
                            particle.particle().mass());
  py8part.vProd(particle.particle().X(), particle.particle().Y(), particle.particle().Z(), particle.particle().T());
  decayer->event.append(py8part);

  int nentries_before = decayer->event.size();
  decayer->particleData.mayDecay(pid,
                                 true);  // switch on the decay of this and only this particle (avoid double decays)
  decayer->next();                       // do the decay
  decayer->particleData.mayDecay(pid, false);  // switch it off again
  int nentries_after = decayer->event.size();
  if (nentries_after <= nentries_before)
    return theList;

  theList.reserve(nentries_after - nentries_before);

  for (int ipart = nentries_before; ipart < nentries_after; ipart++) {
    Pythia8::Particle& py8daughter = decayer->event[ipart];
    theList
        .emplace_back(makeParticle(
            particle.particleDataTable(),
            py8daughter.id(),
            XYZTLorentzVector(py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e()),
            XYZTLorentzVector(py8daughter.xProd(), py8daughter.yProd(), py8daughter.zProd(), py8daughter.tProd())))
        .setMass(py8daughter.m());
  }

  return theList;
}
