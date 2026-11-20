#ifndef FastSimulation_CaloGeometryTools_HGCalFastGeometry_h
#define FastSimulation_CaloGeometryTools_HGCalFastGeometry_h

/** \class HGCalFastGeometry
 *
 * Fast (layer, x, y, zside) -> HGCal DetId lookup for FastSimulation.
 *
 * Motivation: HGCalGeometry::getClosestCell() linearly scans every geometry
 * module of the subdetector (O(1e4) Si wafers, O(1e5) scintillator tiles),
 * calling getPosition() per entry -- and then discards the result, recomputing
 * layer and cell from the position anyway. It was the dominant cost of the 2021
 * HGCal FastSim prototype. FullSim itself never uses it: HGCalNumberingScheme
 * takes the layer from the G4 volume hierarchy and calls
 * HGCalDDDConstants::waferFromPosition().
 *
 * This class removes the scan. At construction it buckets the (cassette-shifted)
 * wafer centres of every layer into a uniform 2D grid. A lookup then bins the
 * point, examines only the 3x3 neighbouring buckets, and picks the nearest wafer
 * centre -- which for a hexagonal lattice is exactly the containing wafer, since
 * the Voronoi cells of a hex lattice are the hexagons themselves. The wafer
 * (u,v) is then handed to waferFromPosition(), which skips its own O(N) wafer
 * search and performs only the analytic cell inversion. Cell assignment is
 * therefore delegated to the very code FullSim uses, so results agree by
 * construction.
 *
 * Memory is bounded and small (one entry per wafer per layer, ~0.35 MB for CE-E
 * and CE-H silicon together) -- deliberately unlike
 * CaloGeometryHelper::buildCrystalArray(), which pre-caches every crystal and
 * would be untenable for ~6M HGCal cells.
 *
 * Units: all public positions are in cm (reco convention). The sim/DDD mm
 * convention and the xx = zside*x flip are handled internally.
 *
 * The object keeps a reference to the HGCalDDDConstants it was built from, so
 * it must not outlive that ESProduct; rebuild it when the geometry changes.
 * All member functions are const, so one instance can be shared across streams.
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"

class HGCalDDDConstants;

class HGCalFastGeometry {
public:
  /// \param ddd    constants for one subdetector (EE, HSi or HSc)
  /// \param det    DetId::HGCalEE, DetId::HGCalHSi or DetId::HGCalHSc
  /// \param binCm  bucket size in cm; 0 (default) means the wafer pitch. Smaller
  ///               values are raised to the pitch, since the 3x3 bucket scan
  ///               only covers the containing wafer for buckets at least that big.
  HGCalFastGeometry(const HGCalDDDConstants& ddd, DetId::Detector det, double binCm = 0.);

  /// Layer number (as stored in the DetId) of the first/last layer of this subdetector.
  int firstLayer() const { return firstLayer_; }
  int lastLayer() const { return lastLayer_; }
  int nLayers() const { return lastLayer_ - firstLayer_ + 1; }

  /// z of a layer, cm, positive side. Returns 0 for an unknown layer.
  double layerZ(int layer) const;

  /// (rMin, rMax) of a layer, cm. Returns (0,0) for an unknown layer.
  std::pair<double, double> layerRadii(int layer) const;

  /// Layer nearest to |z| (cm), or -1 if |z| is more than half a layer spacing
  /// in front of the first or behind the last layer of this subdetector.
  int layerOf(double z) const;

  /// The lookup. Returns a null DetId if the point is not on a valid cell.
  /// \param layer  layer number as stored in the DetId
  /// \param x,y    position in cm, global frame
  /// \param zside  +1 or -1
  DetId toDetId(int layer, double x, double y, int zside) const;

  /// Silicon thickness index of a cell: 0=HD120, 1=LD200, 2=LD300, 3=HD200.
  /// Returns -1 for scintillator. Cheap: decoded straight from the DetId.
  static int siThicknessType(const DetId& id);

  /// The sim-step thickness weight FullSim applies in HGCalSD (HGCalDDDConstants
  /// scales HD120 cells by cellThickness/waferThick when useSimWt is set; every
  /// other cell gets 1.0). FastSim must apply the same weight to match SimHits.
  double simWeight(const DetId& id) const;

  /// Diagnostics.
  bool isSilicon() const { return silicon_; }
  std::size_t nCachedWafers() const { return wafers_.size(); }

private:
  struct Wafer {
    float x, y;    // cassette-shifted centre, cm, +z side
    int16_t u, v;  // wafer indices
  };

  // One uniform bucket grid per layer, stored CSR-style to keep the footprint
  // low (offsets into a single flat wafer array; no per-bucket std::vector).
  struct LayerGrid {
    double xMin = 0., yMin = 0.;
    double binCm = 0.;
    int nx = 0, ny = 0;
    uint32_t waferBegin = 0;  // range in wafers_
    uint32_t waferEnd = 0;
    std::vector<uint32_t> bucketOffset;  // size nx*ny+1, indexes bucketWafer
    std::vector<uint32_t> bucketWafer;   // indices into wafers_
  };

  void buildLayer(int layer, double binCm);
  int nearestWafer(const LayerGrid& g, double x, double y, int& u, int& v) const;

  const HGCalDDDConstants& ddd_;
  const DetId::Detector det_;
  bool silicon_;
  int firstLayer_ = 0, lastLayer_ = -1;

  std::vector<Wafer> wafers_;
  std::vector<LayerGrid> grids_;  // indexed by layer - firstLayer_
  std::vector<double> layerZ_;    // cm, positive side
  std::vector<std::pair<double, double>> layerR_;
};

#endif
