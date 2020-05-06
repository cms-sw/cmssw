#include "RecoTauTag/RecoTau/interface/PositionAtECalEntrance.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

#include <cassert>

PositionAtECalEntrance::PositionAtECalEntrance() : bField_z_(-1.) {}

PositionAtECalEntrance::~PositionAtECalEntrance() {}

void PositionAtECalEntrance::beginEvent(const edm::EventSetup& es) {
  edm::ESHandle<MagneticField> bField;
  es.get<IdealMagneticFieldRecord>().get(bField);
  bField_z_ = bField->inTesla(GlobalPoint(0., 0., 0.)).z();
}

reco::Candidate::Point PositionAtECalEntrance::operator()(const reco::Candidate* particle, bool& success) const {
  assert(bField_z_ != -1.);
  BaseParticlePropagator propagator = BaseParticlePropagator(
      RawParticle(math::XYZTLorentzVector(particle->px(), particle->py(), particle->pz(), particle->energy()),
                  math::XYZTLorentzVector(particle->vertex().x(), particle->vertex().y(), particle->vertex().z(), 0.)),
      0.,
      0.,
      bField_z_);
  propagator.setCharge(particle->charge());
  propagator.propagateToEcalEntrance(false);
  reco::Candidate::Point position;
  if (propagator.getSuccess() != 0) {
    position = reco::Candidate::Point(
        propagator.vertex().x(), propagator.vertex().y(), propagator.vertex().z());
    success = true;
  } else {
    success = false;
  }
  return position;
}
