#include "RecoMuon/MuonIdentification/interface/MuonKinkFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

MuonKinkFinder::MuonKinkFinder(const edm::ParameterSet &iConfig, edm::ConsumesCollector &iC)
    : diagonalOnly_(iConfig.getParameter<bool>("diagonalOnly")),
      usePosition_(iConfig.getParameter<bool>("usePosition")),
      refitter_(iConfig, iC) {}

MuonKinkFinder::~MuonKinkFinder() {}

void MuonKinkFinder::init(const edm::EventSetup &iSetup) { refitter_.setServices(iSetup); }

bool MuonKinkFinder::fillTrkKink(reco::MuonQuality &quality, const reco::Track &track) const {
  std::vector<Trajectory> traj = refitter_.transform(track);
  if (traj.size() != 1) {
    quality.trkKink = 999;
    quality.tkKink_position = math::XYZPoint(0, 0, 0);
    return false;
  }
  return fillTrkKink(quality, traj.front());
}

bool MuonKinkFinder::fillTrkKink(reco::MuonQuality &quality, const Trajectory &trajectory) const {
  const std::vector<TrajectoryMeasurement> &tms = trajectory.measurements();
  quality.trkKink = -1.0;
  quality.tkKink_position = math::XYZPoint(0, 0, 0);
  bool found = false;
  for (int itm = 3, nm = tms.size() - 3; itm < nm; ++itm) {
    TrajectoryStateOnSurface pre = tms[itm].forwardPredictedState();
    TrajectoryStateOnSurface post = tms[itm].backwardPredictedState();
    if (!pre.isValid() || !post.isValid())
      continue;
    found = true;
    double c2f = getChi2(pre, post);
    if (c2f > quality.trkKink) {
      quality.trkKink = c2f;
      GlobalPoint pos = (tms[itm].updatedState().isValid() ? tms[itm].updatedState() : pre).globalPosition();
      quality.tkKink_position = math::XYZPoint(pos.x(), pos.y(), pos.z());
    }
  }
  if (!found)
    quality.trkKink = 999;
  return found;
}

double MuonKinkFinder::getChi2(const TrajectoryStateOnSurface &start, const TrajectoryStateOnSurface &other) const {
  if (!start.hasError() && !other.hasError())
    throw cms::Exception("LogicError") << "At least one of the two states must have errors to make chi2s.\n";
  AlgebraicSymMatrix55 cov;
  if (start.hasError())
    cov += start.localError().matrix();
  if (other.hasError())
    cov += other.localError().matrix();
  cropAndInvert(cov);
  AlgebraicVector5 diff(start.localParameters().mixedFormatVector() - other.localParameters().mixedFormatVector());
  return ROOT::Math::Similarity(diff, cov);
}

void MuonKinkFinder::cropAndInvert(AlgebraicSymMatrix55 &cov) const {
  if (usePosition_) {
    if (diagonalOnly_) {
      for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j < 5; ++j) {
          cov(i, j) = 0;
        }
      }
    }
    cov.Invert();
  } else {
    // get 3x3 covariance
    AlgebraicSymMatrix33 momCov = cov.Sub<AlgebraicSymMatrix33>(0, 0);  // get 3x3 matrix
    if (diagonalOnly_) {
      momCov(0, 1) = 0;
      momCov(0, 2) = 0;
      momCov(1, 2) = 0;
    }
    // invert
    momCov.Invert();
    // place it
    cov.Place_at(momCov, 0, 0);
    // zero the rest
    for (size_t i = 3; i < 5; ++i) {
      for (size_t j = i; j < 5; ++j) {
        cov(i, j) = 0;
      }
    }
  }
}
