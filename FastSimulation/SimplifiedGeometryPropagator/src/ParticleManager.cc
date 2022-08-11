#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleManager.h"

#include "HepMC/GenEvent.h"
#include "HepMC/Units.h"
#include "HepPDT/ParticleDataTable.hh"

#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleFilter.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

fastsim::ParticleManager::ParticleManager(const HepMC::GenEvent& genEvent,
                                          const HepPDT::ParticleDataTable& particleDataTable,
                                          double beamPipeRadius,
                                          double deltaRchargedMother,
                                          const fastsim::ParticleFilter& particleFilter,
                                          std::vector<SimTrack>& simTracks,
                                          std::vector<SimVertex>& simVertices,
                                          bool useFastSimsDecayer)
    : genEvent_(&genEvent),
      genParticleIterator_(genEvent_->particles_begin()),
      genParticleEnd_(genEvent_->particles_end()),
      genParticleIndex_(1),
      particleDataTable_(&particleDataTable),
      beamPipeRadius2_(beamPipeRadius * beamPipeRadius),
      deltaRchargedMother_(deltaRchargedMother),
      particleFilter_(&particleFilter),
      simTracks_(&simTracks),
      simVertices_(&simVertices),
      useFastSimsDecayer_(useFastSimsDecayer)
      // prepare unit convsersions
      //  --------------------------------------------
      // |          |      hepmc               |  cms |
      //  --------------------------------------------
      // | length   | genEvent_->length_unit   |  cm  |
      // | momentum | genEvent_->momentum_unit |  GeV |
      // | time     | length unit (t*c)        |  ns  |
      //  --------------------------------------------
      ,
      momentumUnitConversionFactor_(conversion_factor(genEvent_->momentum_unit(), HepMC::Units::GEV)),
      lengthUnitConversionFactor_(conversion_factor(genEvent_->length_unit(), HepMC::Units::LengthUnit::CM)),
      lengthUnitConversionFactor2_(lengthUnitConversionFactor_ * lengthUnitConversionFactor_),
      timeUnitConversionFactor_(lengthUnitConversionFactor_ / fastsim::Constants::speedOfLight)

{
  // add the main vertex from the signal event to the simvertex collection
  if (genEvent.vertices_begin() != genEvent_->vertices_end()) {
    const HepMC::FourVector& position = (*genEvent.vertices_begin())->position();
    addSimVertex(math::XYZTLorentzVector(position.x() * lengthUnitConversionFactor_,
                                         position.y() * lengthUnitConversionFactor_,
                                         position.z() * lengthUnitConversionFactor_,
                                         position.t() * timeUnitConversionFactor_),
                 -1);
  }
}

fastsim::ParticleManager::~ParticleManager() {}

std::unique_ptr<fastsim::Particle> fastsim::ParticleManager::nextParticle(const RandomEngineAndDistribution& random) {
  std::unique_ptr<fastsim::Particle> particle;

  // retrieve particle from buffer
  if (!particleBuffer_.empty()) {
    particle = std::move(particleBuffer_.back());
    particleBuffer_.pop_back();
  }
  // or from genParticle list
  else {
    particle = nextGenParticle();
    if (!particle)
      return nullptr;
  }

  // if filter does not accept, skip particle
  if (!particleFilter_->accepts(*particle)) {
    return nextParticle(random);
  }

  // lifetime or charge of particle are not yet set
  if (!particle->remainingProperLifeTimeIsSet() || !particle->chargeIsSet()) {
    // retrieve the particle data
    const HepPDT::ParticleData* particleData = particleDataTable_->particle(HepPDT::ParticleID(particle->pdgId()));
    if (!particleData) {
      // in very few events the Decayer (pythia) produces high mass resonances that are for some reason not present in the table (even though they should technically be)
      // they have short lifetimes, so decay them right away (charge and lifetime cannot be taken from table)
      particle->setRemainingProperLifeTimeC(0.);
      particle->setCharge(0.);
    }

    // set lifetime
    if (!particle->remainingProperLifeTimeIsSet()) {
      // The lifetime is 0. in the Pythia Particle Data Table! Calculate from width instead (ct=hbar/width).
      // ct=particleData->lifetime().value();
      double width = particleData->totalWidth().value();
      if (width > 1.0e-35) {
        particle->setRemainingProperLifeTimeC(-log(random.flatShoot()) * 6.582119e-25 / width / 10.);  // ct in cm
      } else {
        particle->setStable();
      }
    }

    // set charge
    if (!particle->chargeIsSet()) {
      particle->setCharge(particleData->charge());
    }
  }

  // add corresponding simTrack to simTrack collection
  unsigned simTrackIndex = addSimTrack(particle.get());
  particle->setSimTrackIndex(simTrackIndex);

  // and return
  return particle;
}

void fastsim::ParticleManager::addSecondaries(const math::XYZTLorentzVector& vertexPosition,
                                              int parentSimTrackIndex,
                                              std::vector<std::unique_ptr<Particle> >& secondaries,
                                              const SimplifiedGeometry* layer) {
  // vertex must be within the accepted volume
  if (!particleFilter_->acceptsVtx(vertexPosition)) {
    return;
  }

  // no need to create vertex in case no particles are produced
  if (secondaries.empty()) {
    return;
  }

  // add simVertex
  unsigned simVertexIndex = addSimVertex(vertexPosition, parentSimTrackIndex);

  // closest charged daughter continues the track of the mother particle
  // simplified tracking algorithm for fastSim
  double distMin = 99999.;
  int idx = -1;
  int idxMin = -1;
  for (auto& secondary : secondaries) {
    idx++;
    if (secondary->getMotherDeltaR() != -1) {
      if (secondary->getMotherDeltaR() > deltaRchargedMother_) {
        // larger than max requirement on deltaR
        secondary->resetMother();
      } else {
        if (secondary->getMotherDeltaR() < distMin) {
          distMin = secondary->getMotherDeltaR();
          idxMin = idx;
        }
      }
    }
  }

  // add secondaries to buffer
  idx = -1;
  for (auto& secondary : secondaries) {
    idx++;
    if (idxMin != -1) {
      // reset all but the particle with the lowest deltaR (which is at idxMin)
      if (secondary->getMotherDeltaR() != -1 && idx != idxMin) {
        secondary->resetMother();
      }
    }

    // set origin vertex
    secondary->setSimVertexIndex(simVertexIndex);
    //
    if (layer) {
      secondary->setOnLayer(layer->isForward(), layer->index());
    }
    // ...and add particle to buffer
    particleBuffer_.push_back(std::move(secondary));
  }
}

unsigned fastsim::ParticleManager::addEndVertex(const fastsim::Particle* particle) {
  return this->addSimVertex(particle->position(), particle->simTrackIndex());
}

unsigned fastsim::ParticleManager::addSimVertex(const math::XYZTLorentzVector& position, int parentSimTrackIndex) {
  int simVertexIndex = simVertices_->size();
  simVertices_->emplace_back(position.Vect(), position.T(), parentSimTrackIndex, simVertexIndex);
  return simVertexIndex;
}

unsigned fastsim::ParticleManager::addSimTrack(const fastsim::Particle* particle) {
  int simTrackIndex = simTracks_->size();
  simTracks_->emplace_back(
      particle->pdgId(), particle->momentum(), particle->simVertexIndex(), particle->genParticleIndex());
  simTracks_->back().setTrackId(simTrackIndex);
  return simTrackIndex;
}

std::unique_ptr<fastsim::Particle> fastsim::ParticleManager::nextGenParticle() {
  // only consider particles that start in the beam pipe and end outside the beam pipe
  // try to get the decay time from pythia
  // use hepmc units
  // make the link simtrack to simvertex
  // try not to change the simvertex structure

  // loop over gen particles
  for (; genParticleIterator_ != genParticleEnd_; ++genParticleIterator_, ++genParticleIndex_) {
    // some handy pointers and references
    const HepMC::GenParticle& particle = **genParticleIterator_;
    const HepMC::GenVertex* productionVertex = particle.production_vertex();
    const HepMC::GenVertex* endVertex = particle.end_vertex();

    // skip incoming particles
    if (!productionVertex) {
      continue;
    }
    if (std::abs(particle.pdg_id()) < 10 || std::abs(particle.pdg_id()) == 21) {
      continue;
    }
    // particles which do not descend from exotics must be produced within the beampipe
    int exoticRelativeId = 0;
    const bool producedWithinBeamPipe =
        productionVertex->position().perp2() * lengthUnitConversionFactor2_ < beamPipeRadius2_;
    if (!producedWithinBeamPipe && useFastSimsDecayer_) {
      exoticRelativesChecker(productionVertex, exoticRelativeId, 0);
      if (!isExotic(exoticRelativeId)) {
        continue;
      }
    }

    // FastSim will not make hits out of particles that decay before reaching the beam pipe
    const bool decayedWithinBeamPipe =
        endVertex && endVertex->position().perp2() * lengthUnitConversionFactor2_ < beamPipeRadius2_;
    if (decayedWithinBeamPipe) {
      continue;
    }

    // SM particles that descend from exotics and cross the beam pipe radius should make hits but not be decayed
    if (producedWithinBeamPipe && !decayedWithinBeamPipe && useFastSimsDecayer_) {
      exoticRelativesChecker(productionVertex, exoticRelativeId, 0);
    }

    // make the particle
    std::unique_ptr<Particle> newParticle(
        new Particle(particle.pdg_id(),
                     math::XYZTLorentzVector(productionVertex->position().x() * lengthUnitConversionFactor_,
                                             productionVertex->position().y() * lengthUnitConversionFactor_,
                                             productionVertex->position().z() * lengthUnitConversionFactor_,
                                             productionVertex->position().t() * timeUnitConversionFactor_),
                     math::XYZTLorentzVector(particle.momentum().x() * momentumUnitConversionFactor_,
                                             particle.momentum().y() * momentumUnitConversionFactor_,
                                             particle.momentum().z() * momentumUnitConversionFactor_,
                                             particle.momentum().e() * momentumUnitConversionFactor_)));
    newParticle->setGenParticleIndex(genParticleIndex_);
    if (isExotic(exoticRelativeId)) {
      newParticle->setMotherPdgId(exoticRelativeId);
    }

    // try to get the life time of the particle from the genEvent
    if (endVertex) {
      double labFrameLifeTime =
          (endVertex->position().t() - productionVertex->position().t()) * timeUnitConversionFactor_;
      newParticle->setRemainingProperLifeTimeC(labFrameLifeTime / newParticle->gamma() *
                                               fastsim::Constants::speedOfLight);
    }

    // Find production vertex if it already exists. Otherwise create new vertex
    // Possible to recreate the whole GenEvent using SimTracks/SimVertices (see FBaseSimEvent::fill(..))
    bool foundVtx = false;
    for (const auto& simVtx : *simVertices_) {
      if (std::abs(simVtx.position().x() - newParticle->position().x()) < 1E-3 &&
          std::abs(simVtx.position().y() - newParticle->position().y()) < 1E-3 &&
          std::abs(simVtx.position().z() - newParticle->position().z()) < 1E-3) {
        newParticle->setSimVertexIndex(simVtx.vertexId());
        foundVtx = true;
        break;
      }
    }
    if (!foundVtx)
      newParticle->setSimVertexIndex(addSimVertex(newParticle->position(), -1));

    // iterator/index has to be increased in case of return (is not done by the loop then)
    ++genParticleIterator_;
    ++genParticleIndex_;
    // and return
    return newParticle;
  }

  return std::unique_ptr<Particle>();
}

void fastsim::ParticleManager::exoticRelativesChecker(const HepMC::GenVertex* originVertex,
                                                      int& exoticRelativeId_,
                                                      int ngendepth = 0) {
  if (ngendepth > 99 || exoticRelativeId_ == -1 || isExotic(std::abs(exoticRelativeId_)))
    return;
  ngendepth += 1;
  std::vector<HepMC::GenParticle*>::const_iterator relativesIterator_ = originVertex->particles_in_const_begin();
  std::vector<HepMC::GenParticle*>::const_iterator relativesIteratorEnd_ = originVertex->particles_in_const_end();
  for (; relativesIterator_ != relativesIteratorEnd_; ++relativesIterator_) {
    const HepMC::GenParticle& genRelative = **relativesIterator_;
    if (isExotic(std::abs(genRelative.pdg_id()))) {
      exoticRelativeId_ = genRelative.pdg_id();
      if (ngendepth == 100)
        exoticRelativeId_ = -1;
      return;
    }
    const HepMC::GenVertex* vertex_ = genRelative.production_vertex();
    if (!vertex_)
      return;
    exoticRelativesChecker(vertex_, exoticRelativeId_, ngendepth);
  }
  return;
}
