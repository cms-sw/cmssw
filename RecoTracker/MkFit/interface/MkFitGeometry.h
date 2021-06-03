#ifndef RecoTracker_MkFit_MkFitGeometry_h
#define RecoTracker_MkFit_MkFitGeometry_h

#include "DataFormats/DetId/interface/DetId.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace mkfit {
  class LayerNumberConverter;
  class TrackerInfo;
  class IterationsInfo;
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
                         std::unique_ptr<mkfit::IterationsInfo> iterationsInfo);
  ~MkFitGeometry();

  int mkFitLayerNumber(DetId detId) const;
  mkfit::LayerNumberConverter const& layerNumberConverter() const { return *lnc_; }
  mkfit::TrackerInfo const& trackerInfo() const { return *trackerInfo_; }
  mkfit::IterationsInfo const& iterationsInfo() const { return *iterationsInfo_; }
  const std::vector<const DetLayer*>& detLayers() const { return dets_; }
  unsigned int uniqueIdInLayer(int layer, unsigned int detId) const { return detIdToShortId_.at(layer).at(detId); }

private:
  const TrackerTopology* ttopo_;
  std::unique_ptr<mkfit::LayerNumberConverter> lnc_;  // for pimpl pattern
  std::unique_ptr<mkfit::TrackerInfo> trackerInfo_;
  std::unique_ptr<mkfit::IterationsInfo> iterationsInfo_;  // only temporarily here, to be moved into proper place later
  std::vector<const DetLayer*> dets_;
  std::vector<std::unordered_map<unsigned int, unsigned int>> detIdToShortId_;
};

#endif
