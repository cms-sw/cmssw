#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"

// mkFit includes
#include "Hit.h"
#include "mkFit/HitStructures.h"

MkFitHitWrapper::MkFitHitWrapper() = default;
MkFitHitWrapper::MkFitHitWrapper(mkfit::TrackerInfo const& trackerInfo)
    : eventOfHits_(std::make_unique<mkfit::EventOfHits>(trackerInfo)),
      pixelHits_(std::make_unique<mkfit::HitVec>()),
      outerHits_(std::make_unique<mkfit::HitVec>()) {}

MkFitHitWrapper::~MkFitHitWrapper() = default;

MkFitHitWrapper::MkFitHitWrapper(MkFitHitWrapper&&) = default;
MkFitHitWrapper& MkFitHitWrapper::operator=(MkFitHitWrapper&&) = default;

void MkFitHitWrapper::stripClusterChargeCut(float minThreshold, std::vector<bool>& mask) const {
  if (mask.size() != stripClusterCharge_.size()) {
    cms::Exception e("LogicError");
    e << "Mask size (" << mask.size() << ") inconsistent with number of hits (" << stripClusterCharge_.size() << ")";
    e.addContext("Calling MkFitHitWraper::applyStripClusterCharge()");
    throw e;
  }
  for (int i = 0, end = stripClusterCharge_.size(); i < end; ++i) {
    // mask == true means skip the cluster
    mask[i] = mask[i] || (stripClusterCharge_[i] <= minThreshold);
  }
}
