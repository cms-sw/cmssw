#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace {
  bool isPlusSide(const TrackerTopology& ttopo, DetId detid) {
    return ttopo.side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap);
  }
}  // namespace

MkFitGeometry::MkFitGeometry(const TrackerGeometry& geom,
                             const GeometricSearchTracker& tracker,
                             const TrackerTopology& ttopo,
                             std::unique_ptr<mkfit::TrackerInfo> trackerInfo,
                             const mkfit::LayerNumberConverter& layNConv)
    : ttopo_(&ttopo),
      lnc_{std::make_unique<mkfit::LayerNumberConverter>(layNConv)},
      trackerInfo_(std::move(trackerInfo)) {
  if (lnc_->getEra() != mkfit::TkLayout::phase1 && lnc_->getEra() != mkfit::TkLayout::phase2)
    throw cms::Exception("Assert") << "This code works only with phase1 and phase2 tracker, you have something else";

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

    setDet(subdet, layer, monoLayer, detId, lay);
    if (lnc_->doesHaveStereo(subdet, layer))
      setDet(subdet, layer, stereoLayer, detId, lay);
  }
}

// Explicit out-of-line because of the mkfit::LayerNumberConverter is
// only forward declared in the header
MkFitGeometry::~MkFitGeometry() {}

int MkFitGeometry::mkFitLayerNumber(DetId detId) const {
  return lnc_->convertLayerNumber(
      detId.subdetId(), ttopo_->layer(detId), false, ttopo_->isStereo(detId), isPlusSide(*ttopo_, detId));
}
