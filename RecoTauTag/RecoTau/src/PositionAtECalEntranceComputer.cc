#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include <cassert>

PositionAtECalEntranceComputer::PositionAtECalEntranceComputer(edm::ConsumesCollector&& cc, bool isPhase2)
    : bField_esToken_(cc.esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      caloGeo_esToken_(cc.esConsumes<CaloGeometry, CaloGeometryRecord>()),
      bField_z_(-1.),
      isPhase2_(isPhase2) {}

PositionAtECalEntranceComputer::PositionAtECalEntranceComputer(edm::ConsumesCollector& cc, bool isPhase2)
    : bField_esToken_(cc.esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      caloGeo_esToken_(cc.esConsumes<CaloGeometry, CaloGeometryRecord>()),
      bField_z_(-1.),
      isPhase2_(isPhase2) {}

PositionAtECalEntranceComputer::~PositionAtECalEntranceComputer() {}

void PositionAtECalEntranceComputer::beginEvent(const edm::EventSetup& es) {
  bField_z_ = es.getData(bField_esToken_).inTesla(GlobalPoint(0., 0., 0.)).z();
  if (isPhase2_) {
    recHitTools_.setGeometry(es.getData(caloGeo_esToken_));
    hgcalFace_z_ = recHitTools_.getPositionLayer(1).z();  // HGCal 1st layer
  }
}

reco::Candidate::Point PositionAtECalEntranceComputer::operator()(const reco::Candidate* particle,
                                                                  bool& success) const {
  assert(bField_z_ != -1.);
  reco::Candidate::Point position;
  BaseParticlePropagator propagator = BaseParticlePropagator(
      RawParticle(particle->p4(),
                  math::XYZTLorentzVector(particle->vertex().x(), particle->vertex().y(), particle->vertex().z(), 0.),
                  particle->charge()),
      0.,
      0.,
      bField_z_);
  if (!isPhase2_ || std::abs(particle->eta()) < ecalBarrelEndcapEtaBorder_) {  // ECal
    propagator.propagateToEcalEntrance(false);
  } else {  // HGCal
    if (std::abs(particle->vertex().z()) >= hgcalFace_z_) {
      success = false;
      return position;
    }
    propagator.setPropagationConditions(152.6, hgcalFace_z_, false);
    propagator.propagate();
  }
  if (propagator.getSuccess() != 0) {
    position = propagator.particle().vertex().Vect();
    success = (std::abs(position.eta()) <= hgcalHfEtaBorder_);
  } else {
    success = false;
  }
  return position;
}
