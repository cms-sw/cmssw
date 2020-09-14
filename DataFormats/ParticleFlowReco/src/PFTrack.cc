#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"
#include "Math/GenVector/PositionVector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;
using namespace std;

const unsigned PFTrack::nMaxTrackingLayers_ = 17;

PFTrack::PFTrack() : charge_(0.), indexInnermost_(0), indexOutermost_(0) {
  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.reserve(PFTrajectoryPoint::NLayers + nMaxTrackingLayers_);
}

PFTrack::PFTrack(double charge) : charge_(charge), indexInnermost_(0), indexOutermost_(0) {
  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.reserve(PFTrajectoryPoint::NLayers + nMaxTrackingLayers_);
}

PFTrack::PFTrack(const PFTrack& other)
    : charge_(other.charge_),
      trajectoryPoints_(other.trajectoryPoints_),
      indexInnermost_(other.indexInnermost_),
      indexOutermost_(other.indexOutermost_) {}

void PFTrack::addPoint(const PFTrajectoryPoint& trajPt) {
  if (trajPt.isTrackerLayer()) {
    if (!indexOutermost_) {  // first time a measurement is added
      if (trajectoryPoints_.size() < PFTrajectoryPoint::BeamPipeOrEndVertex + 1) {
        PFTrajectoryPoint dummyPt;
        for (unsigned iPt = trajectoryPoints_.size(); iPt < PFTrajectoryPoint::BeamPipeOrEndVertex + 1; iPt++)
          trajectoryPoints_.push_back(dummyPt);
      } else if (trajectoryPoints_.size() > PFTrajectoryPoint::BeamPipeOrEndVertex + 1) {
        edm::LogWarning("PFTrack") << "trajectoryPoints_.size() is too large = " << trajectoryPoints_.size();
      }
      indexOutermost_ = indexInnermost_ = PFTrajectoryPoint::BeamPipeOrEndVertex + 1;
    } else
      indexOutermost_++;
  }
  // Use push_back instead of insert in order to gain time
  trajectoryPoints_.push_back(trajPt);
}

const reco::PFTrajectoryPoint& PFTrack::extrapolatedPoint(unsigned layerid) const {
  const unsigned offset_layerid = nTrajectoryMeasurements() + layerid;
  if (layerid >= reco::PFTrajectoryPoint::NLayers || offset_layerid >= trajectoryPoints_.size()) {
    throw cms::Exception("SizeError") << "PFRecTrack::extrapolatedPoint: cannot access " << layerid
                                      << " #traj meas = " << nTrajectoryMeasurements()
                                      << " #traj points = " << trajectoryPoints_.size() << endl
                                      << (*this);
  }
  if (layerid < indexInnermost_)
    return trajectoryPoints_[layerid];
  else
    return trajectoryPoints_[offset_layerid];
}

ostream& reco::operator<<(ostream& out, const PFTrack& track) {
  if (!out)
    return out;

  const reco::PFTrajectoryPoint& closestApproach = track.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out << "Track charge = " << track.charge() << ", Pt = " << closestApproach.momentum().Pt()
      << ", P = " << closestApproach.momentum().P() << endl
      << "\tR0 = " << closestApproach.position().Rho() << " Z0 = " << closestApproach.position().Z() << endl
      << "\tnumber of tracker measurements = " << track.nTrajectoryMeasurements() << endl;
  for (unsigned i = 0; i < track.trajectoryPoints().size(); i++)
    out << track.trajectoryPoints()[i] << endl;

  return out;
}
