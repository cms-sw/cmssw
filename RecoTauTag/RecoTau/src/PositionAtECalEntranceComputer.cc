#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include <cassert>

PositionAtECalEntranceComputer::PositionAtECalEntranceComputer() : bField_z_(-1.) {}

PositionAtECalEntranceComputer::~PositionAtECalEntranceComputer() {}

void PositionAtECalEntranceComputer::beginEvent(const edm::EventSetup& es) {
  edm::ESHandle<MagneticField> bField;
  es.get<IdealMagneticFieldRecord>().get(bField);
  bField_z_ = bField->inTesla(GlobalPoint(0., 0., 0.)).z();
}

reco::Candidate::Point PositionAtECalEntranceComputer::operator()(const reco::Candidate* particle,
                                                                  bool& success) const {
  assert(bField_z_ != -1.);
  BaseParticlePropagator propagator = BaseParticlePropagator(
      RawParticle(particle->p4(),
                  math::XYZTLorentzVector(particle->vertex().x(), particle->vertex().y(), particle->vertex().z(), 0.),
                  particle->charge()),
      0.,
      0.,
      bField_z_);
  propagator.propagateToEcalEntrance(false);
  reco::Candidate::Point position;
  if (propagator.getSuccess() != 0) {
    position = propagator.particle().vertex().Vect();
    success = true;
  } else {
    success = false;
  }
  return position;
}
