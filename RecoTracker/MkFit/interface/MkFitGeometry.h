#ifndef RecoTracker_MkFit_MkFitGeometry_h
#define RecoTracker_MkFit_MkFitGeometry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace mkfit {
  class LayerNumberConverter;
}  // namespace mkfit

class DetLayer;
class GeometricSearchTracker;
class TrackerGeometry;
class TrackerTopology;

/**
 * Collection of geometry-related objects for mkFit
 */
class MkFitGeometry {
public:
  explicit MkFitGeometry(const TrackerGeometry& geom,
                         const GeometricSearchTracker& tracker,
                         const TrackerTopology& ttopo,
                         std::unique_ptr<mkfit::TrackerInfo> trackerInfo,
                         const mkfit::LayerNumberConverter& layNConv);
  ~MkFitGeometry();

  int mkFitLayerNumber(DetId detId) const;
  mkfit::LayerNumberConverter const& layerNumberConverter() const { return *lnc_; }
  mkfit::TrackerInfo const& trackerInfo() const { return *trackerInfo_; }
  const std::vector<const DetLayer*>& detLayers() const { return dets_; }
  unsigned int uniqueIdInLayer(int layer, unsigned int detId) const {
    return trackerInfo_->layer(layer).short_id(detId);
  }
  const TrackerTopology* topology() const { return ttopo_; }

private:
  const TrackerTopology* ttopo_;
  std::unique_ptr<mkfit::LayerNumberConverter> lnc_;  // for pimpl pattern
  std::unique_ptr<mkfit::TrackerInfo> trackerInfo_;
  std::vector<const DetLayer*> dets_;
};

#endif
