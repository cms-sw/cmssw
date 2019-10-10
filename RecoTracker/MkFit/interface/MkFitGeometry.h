#ifndef RecoTracker_MkFit_MkFitGeometry_h
#define RecoTracker_MkFit_MkFitGeometry_h

#include <memory>
#include <unordered_map>
#include <vector>

namespace mkfit {
  class LayerNumberConverter;
}

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
                         const TrackerTopology& ttopo);
  ~MkFitGeometry();

  mkfit::LayerNumberConverter const& layerNumberConverter() const { return *lnc_; }
  const std::vector<const DetLayer*>& detLayers() const { return dets_; }
  unsigned int uniqueIdInLayer(int layer, unsigned int detId) const { return detIdToShortId_.at(layer).at(detId); }

private:
  std::unique_ptr<mkfit::LayerNumberConverter> lnc_;  // for pimpl pattern
  std::vector<const DetLayer*> dets_;
  std::vector<std::unordered_map<unsigned int, unsigned int>> detIdToShortId_;
};

#endif
