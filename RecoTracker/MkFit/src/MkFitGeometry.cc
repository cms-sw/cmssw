#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

#include "LayerNumberConverter.h"
#include "TrackerInfo.h"
#include "mkFit/IterationConfig.h"

namespace {
  bool isPlusSide(const TrackerTopology& ttopo, DetId detid) {
    return ttopo.side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap);
  }
}  // namespace

MkFitGeometry::MkFitGeometry(const TrackerGeometry& geom,
                             const GeometricSearchTracker& tracker,
                             const TrackerTopology& ttopo,
                             std::unique_ptr<mkfit::TrackerInfo> trackerInfo,
                             std::unique_ptr<mkfit::IterationsInfo> iterationsInfo)
    : ttopo_(&ttopo),
      lnc_{std::make_unique<mkfit::LayerNumberConverter>(mkfit::TkLayout::phase1)},
      trackerInfo_(std::move(trackerInfo)),
      iterationsInfo_(std::move(iterationsInfo)) {
  if (geom.numberOfLayers(PixelSubdetector::PixelBarrel) != 4 ||
      geom.numberOfLayers(PixelSubdetector::PixelEndcap) != 3) {
    throw cms::Exception("Assert") << "For now this code works only with phase1 tracker, you have something else";
  }

  // Create DetLayer structure
  dets_.resize(lnc_->nLayers(), nullptr);
  auto setDet = [this](const int subdet, const int layer, const int isStereo, const DetId& detId, const DetLayer* lay) {
    const int index = lnc_->convertLayerNumber(subdet, layer, false, isStereo, isPlusSide(*ttopo_, detId));
    if (index < 0 or static_cast<unsigned>(index) >= dets_.size()) {
      throw cms::Exception("LogicError") << "Invalid mkFit layer index " << index << " for DetId " << detId.rawId()
                                         << " subdet " << subdet << " layer " << layer << " isStereo " << isStereo;
    }
    dets_[index] = lay;
  };
  constexpr int monoLayer = 0;
  constexpr int stereoLayer = 1;
  for (const DetLayer* lay : tracker.allLayers()) {
    const auto& comp = lay->basicComponents();
    if (UNLIKELY(comp.empty())) {
      throw cms::Exception("LogicError") << "Got a tracker layer (subdet " << lay->subDetector()
                                         << ") with empty basicComponents.";
    }
    // First component is enough for layer and side information
    const auto& detId = comp.front()->geographicalId();
    const auto subdet = detId.subdetId();
    const auto layer = ttopo.layer(detId);

    // TODO: mono/stereo structure is still hardcoded for phase0/1 strip tracker
    setDet(subdet, layer, monoLayer, detId, lay);
    if (((subdet == StripSubdetector::TIB or subdet == StripSubdetector::TOB) and (layer == 1 or layer == 2)) or
        subdet == StripSubdetector::TID or subdet == StripSubdetector::TEC) {
      setDet(subdet, layer, stereoLayer, detId, lay);
    }
  }

  // Create "short id" aka "unique id within layer"
  detIdToShortId_.resize(lnc_->nLayers());
  for (const auto& detId : geom.detIds()) {
    const auto ilay = mkFitLayerNumber(detId);
    auto& map = detIdToShortId_[ilay];
    const unsigned int ind = map.size();
    // Make sure the short id fits in the 12 bits...
    assert(ind < (int)1 << 11);
    map[detId.rawId()] = ind;
  }
}

// Explicit out-of-line because of the mkfit::LayerNumberConverter is
// only forward declared in the header
MkFitGeometry::~MkFitGeometry() {}

int MkFitGeometry::mkFitLayerNumber(DetId detId) const {
  return lnc_->convertLayerNumber(
      detId.subdetId(), ttopo_->layer(detId), false, ttopo_->isStereo(detId), isPlusSide(*ttopo_, detId));
}
