#include "FastSimulation/SimplifiedGeometryPropagator/interface/Decayer.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleManager.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"
#include "GeneratorInterface/Pythia8Interface/interface/P8RndmEngine.h"

#include <Pythia8/Pythia.h>
#include "Pythia8Plugins/HepMC2.h"

fastsim::Decayer::~Decayer() { ; }

fastsim::Decayer::Decayer() : pythia_(new Pythia8::Pythia()), pythiaRandomEngine_(new gen::P8RndmEngine()) {
  pythia_->setRndmEnginePtr(pythiaRandomEngine_.get());
  pythia_->settings.flag("ProcessLevel:all", false);
  pythia_->settings.flag("PartonLevel:FSRinResonances", false);
  pythia_->settings.flag("ProcessLevel:resonanceDecays", false);
  pythia_->init();

  // forbid all decays
  // (decays are allowed selectively in the decay function)
  Pythia8::ParticleData& pdt = pythia_->particleData;
  int pid = 0;
  while (pdt.nextId(pid) > pid) {
    pid = pdt.nextId(pid);
    pdt.mayDecay(pid, false);
  }
}

void fastsim::Decayer::decay(const Particle& particle,
                             std::vector<std::unique_ptr<fastsim::Particle> >& secondaries,
                             CLHEP::HepRandomEngine& engine) const {
  // make sure pythia takes random numbers from the engine past through via the function arguments
  edm::RandomEngineSentry<gen::P8RndmEngine> sentry(pythiaRandomEngine_.get(), &engine);

  // inspired by method Pythia8Hadronizer::residualDecay() in GeneratorInterface/Pythia8Interface/src/Py8GunBase.cc
  int pid = particle.pdgId();
  // snip decay products of exotic particles or their children. These decay products are preserved from the event record.
  // limitation: if exotic incurs heavy energy loss during propagation, the saved decay products could be too hard.

  if (isExotic(pid) || isExotic(particle.getMotherPdgId())) {
    return;
  }

  pythia_->event.reset();

  // create a pythia particle which has the same properties as the FastSim particle
  Pythia8::Particle pythiaParticle(pid,
                                   93,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   particle.momentum().X(),
                                   particle.momentum().Y(),
                                   particle.momentum().Z(),
                                   particle.momentum().E(),
                                   particle.momentum().M());
  pythiaParticle.vProd(
      particle.position().X(), particle.position().Y(), particle.position().Z(), particle.position().T());
  pythia_->event.append(pythiaParticle);

  int nentries_before = pythia_->event.size();
  // switch on the decay of this and only this particle (avoid double decays)
  pythia_->particleData.mayDecay(pid, true);
  // do the decay
  pythia_->next();
  // switch it off again
  pythia_->particleData.mayDecay(pid, false);
  int nentries_after = pythia_->event.size();

  if (nentries_after <= nentries_before)
    return;

  // add decay products back to the event
  for (int ipart = nentries_before; ipart < nentries_after; ipart++) {
    Pythia8::Particle& daughter = pythia_->event[ipart];

    secondaries.emplace_back(new fastsim::Particle(
        daughter.id(),
        math::XYZTLorentzVector(daughter.xProd(), daughter.yProd(), daughter.zProd(), daughter.tProd()),
        math::XYZTLorentzVector(daughter.px(), daughter.py(), daughter.pz(), daughter.e())));

    // daughter can inherit the SimTrackIndex of mother (if both charged): necessary for FastSim (cheat) tracking
    if (particle.charge() != 0 && std::abs(particle.charge() - daughter.charge()) < 1E-10) {
      secondaries.back()->setMotherDeltaR(particle.momentum());
      secondaries.back()->setMotherPdgId(particle.getMotherDeltaR() == -1 ? particle.pdgId()
                                                                          : particle.getMotherPdgId());
      secondaries.back()->setMotherSimTrackIndex(particle.simTrackIndex());
    }
  }

  return;
}
