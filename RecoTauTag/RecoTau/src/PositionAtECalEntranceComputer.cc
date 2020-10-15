#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"

#include <cassert>

//HGCal helper classes
//MB: looks be copy of HGCal utils: L1Trigger/L1THGCalUtilities/plugins/ntuples/HGCalTriggerNtupleGen.cc
namespace hgcal_helpers {
  class SimpleTrackPropagator {
  public:
    SimpleTrackPropagator(MagneticField const* f) : field_(f), prod_(field_, alongMomentum), absz_target_(0) {
      ROOT::Math::SMatrixIdentity id;
      AlgebraicSymMatrix55 C(id);
      //MB: To define uncertainty of starting point of trajectory propagation scale identity matrix created above by 0.001
      C *= 0.001;
      err_ = CurvilinearTrajectoryError(C);
    }
    void setPropagationTargetZ(const float& z);

    bool propagate(const double& px,
                   const double& py,
                   const double& pz,
                   const double& x,
                   const double& y,
                   const double& z,
                   const float& charge,
                   reco::Candidate::Point& output) const;

  private:
    SimpleTrackPropagator() : field_(nullptr), prod_(field_, alongMomentum), absz_target_(0) {}
    const RKPropagatorInS& RKProp() const { return prod_.propagator; }
    Plane::PlanePointer targetPlaneForward_, targetPlaneBackward_;
    MagneticField const* field_;
    CurvilinearTrajectoryError err_;
    defaultRKPropagator::Product prod_;
    float absz_target_;
  };
  void SimpleTrackPropagator::setPropagationTargetZ(const float& z) {
    targetPlaneForward_ = Plane::build(Plane::PositionType(0, 0, std::abs(z)), Plane::RotationType());
    targetPlaneBackward_ = Plane::build(Plane::PositionType(0, 0, -std::abs(z)), Plane::RotationType());
    absz_target_ = std::abs(z);
  }
  bool SimpleTrackPropagator::propagate(const double& px,
                                        const double& py,
                                        const double& pz,
                                        const double& x,
                                        const double& y,
                                        const double& z,
                                        const float& charge,
                                        reco::Candidate::Point& output) const {
    typedef TrajectoryStateOnSurface TSOS;
    GlobalPoint startingPosition(x, y, z);
    GlobalVector startingMomentum(px, py, pz);
    Plane::PlanePointer startingPlane = Plane::build(Plane::PositionType(x, y, z), Plane::RotationType());
    TSOS startingStateP(
        GlobalTrajectoryParameters(startingPosition, startingMomentum, charge, field_), err_, *startingPlane);
    TSOS trackStateP;
    if (pz > 0) {
      trackStateP = RKProp().propagate(startingStateP, *targetPlaneForward_);
    } else {
      trackStateP = RKProp().propagate(startingStateP, *targetPlaneBackward_);
    }
    if (trackStateP.isValid()) {
      output.SetXYZ(
          trackStateP.globalPosition().x(), trackStateP.globalPosition().y(), trackStateP.globalPosition().z());
      return true;
    }
    output.SetXYZ(0, 0, 0);
    return false;
  }
}  // namespace hgcal_helpers

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
  bField_ = &es.getData(bField_esToken_);
  bField_z_ = bField_->inTesla(GlobalPoint(0., 0., 0.)).z();
  if (isPhase2_) {
    recHitTools_.setGeometry(es.getData(caloGeo_esToken_));
    hgcalFace_z_ = recHitTools_.getPositionLayer(1).z();  // HGCal 1st layer
  }
}

reco::Candidate::Point PositionAtECalEntranceComputer::operator()(const reco::Candidate* particle,
                                                                  bool& success) const {
  assert(bField_z_ != -1.);
  reco::Candidate::Point position;
  if (!isPhase2_ || std::abs(particle->eta()) < ecalBarrelEndcapEtaBorder_) {  // ECal
    BaseParticlePropagator propagator = BaseParticlePropagator(
        RawParticle(particle->p4(),
                    math::XYZTLorentzVector(particle->vertex().x(), particle->vertex().y(), particle->vertex().z(), 0.),
                    particle->charge()),
        0.,
        0.,
        bField_z_);
    propagator.propagateToEcalEntrance(false);
    if (propagator.getSuccess() != 0) {
      position = propagator.particle().vertex().Vect();
      success = true;
    } else {
      success = false;
    }
  } else {  // HGCal
    success = false;
    if (std::abs(particle->vertex().z()) >= hgcalFace_z_)
      return position;

    hgcal_helpers::SimpleTrackPropagator propagator(bField_);
    propagator.setPropagationTargetZ(hgcalFace_z_);
    success = propagator.propagate(particle->px(),
                                   particle->py(),
                                   particle->pz(),
                                   particle->vertex().x(),
                                   particle->vertex().y(),
                                   particle->vertex().z(),
                                   particle->charge(),
                                   position);
  }
  return position;
}
