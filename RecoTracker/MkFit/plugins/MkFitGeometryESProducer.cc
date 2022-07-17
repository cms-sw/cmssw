#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

// mkFit includes
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/ConfigWrapper.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"

#include <sstream>

//------------------------------------------------------------------------------

class MkFitGeometryESProducer : public edm::ESProducer {
public:
  MkFitGeometryESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<MkFitGeometry> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  struct GapCollector {
    struct Interval {
      float x, y;
    };

    void reset_current() { m_current = {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}; }
    void extend_current(float q) {
      m_current.x = std::min(m_current.x, q);
      m_current.y = std::max(m_current.y, q);
    }
    void add_current() { add_interval(m_current.x, m_current.y); }

    void add_interval(float x, float y);

    void sqrt_elements();
    bool find_gap(Interval &itvl, float eps);
    void print_gaps(std::ostream &ostr);

    std::list<Interval> m_coverage;
    Interval m_current;
  };
  typedef std::unordered_map<int, GapCollector> layer_gap_map_t;

  void considerPoint(const GlobalPoint &gp, mkfit::LayerInfo &lay_info);
  void fillShapeAndPlacement(const GeomDet *det, mkfit::TrackerInfo &trk_info, layer_gap_map_t *lgc_map = nullptr);
  void addPixBGeometry(mkfit::TrackerInfo &trk_info);
  void addPixEGeometry(mkfit::TrackerInfo &trk_info);
  void addTIBGeometry(mkfit::TrackerInfo &trk_info);
  void addTOBGeometry(mkfit::TrackerInfo &trk_info);
  void addTIDGeometry(mkfit::TrackerInfo &trk_info);
  void addTECGeometry(mkfit::TrackerInfo &trk_info);

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;

  const TrackerTopology *trackerTopo_ = nullptr;
  const TrackerGeometry *trackerGeom_ = nullptr;
  mkfit::LayerNumberConverter layerNrConv_ = {mkfit::TkLayout::phase1};
};

MkFitGeometryESProducer::MkFitGeometryESProducer(const edm::ParameterSet &iConfig) {
  auto cc = setWhatProduced(this);
  geomToken_ = cc.consumes();
  ttopoToken_ = cc.consumes();
  trackerToken_ = cc.consumes();
}

void MkFitGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

//------------------------------------------------------------------------------

void MkFitGeometryESProducer::GapCollector::add_interval(float x, float y) {
  if (x > y)
    std::swap(x, y);
  bool handled = false;
  for (auto i = m_coverage.begin(); i != m_coverage.end(); ++i) {
    if (y < i->x) {  // fully on 'left'
      m_coverage.insert(i, {x, y});
      handled = true;
      break;
    } else if (x > i->y) {  // fully on 'right'
      continue;
    } else if (x < i->x) {  // sticking out on 'left'
      i->x = x;
      handled = true;
      break;
    } else if (y > i->y) {  // sticking out on 'right'
      i->y = y;
      // check for overlap with the next interval, potentially merge
      auto j = i;
      ++j;
      if (j != m_coverage.end() && i->y >= j->x) {
        i->y = j->y;
        m_coverage.erase(j);
      }
      handled = true;
      break;
    } else {  // contained in current interval
      handled = true;
      break;
    }
  }
  if (!handled) {
    m_coverage.push_back({x, y});
  }
}

void MkFitGeometryESProducer::GapCollector::sqrt_elements() {
  for (auto &itvl : m_coverage) {
    itvl.x = std::sqrt(itvl.x);
    itvl.y = std::sqrt(itvl.y);
  }
}

bool MkFitGeometryESProducer::GapCollector::find_gap(Interval &itvl, float eps) {
  auto i = m_coverage.begin();
  while (i != m_coverage.end()) {
    auto j = i;
    ++j;
    if (j != m_coverage.end()) {
      if (j->x - i->y > eps) {
        itvl = {i->y, j->x};
        return true;
      }
      i = j;
    } else {
      break;
    }
  }
  return false;
}

void MkFitGeometryESProducer::GapCollector::print_gaps(std::ostream &ostr) {
  auto i = m_coverage.begin();
  while (i != m_coverage.end()) {
    auto j = i;
    ++j;
    if (j != m_coverage.end()) {
      ostr << "(" << i->y << ", " << j->x << ")->" << j->x - i->y;
      i = j;
    } else {
      break;
    }
  }
}

//------------------------------------------------------------------------------

void MkFitGeometryESProducer::considerPoint(const GlobalPoint &gp, mkfit::LayerInfo &li) {
  // Use radius squared during bounding-region search.
  float r = gp.perp2(), z = gp.z();
  li.extend_limits(r, z);
}

void MkFitGeometryESProducer::fillShapeAndPlacement(const GeomDet *det,
                                                    mkfit::TrackerInfo &trk_info,
                                                    layer_gap_map_t *lgc_map) {
  DetId detid = det->geographicalId();

  float xy[4][2];
  float dz;
  const Bounds *b = &((det->surface()).bounds());

  if (const TrapezoidalPlaneBounds *b2 = dynamic_cast<const TrapezoidalPlaneBounds *>(b)) {
    // See sec. "TrapezoidalPlaneBounds parameters" in doc/reco-geom-notes.txt
    std::array<const float, 4> const &par = b2->parameters();
    xy[0][0] = -par[0];
    xy[0][1] = -par[3];
    xy[1][0] = -par[1];
    xy[1][1] = par[3];
    xy[2][0] = par[1];
    xy[2][1] = par[3];
    xy[3][0] = par[0];
    xy[3][1] = -par[3];
    dz = par[2];

    // printf("TRAP 0x%x %f %f %f %f\n", detid.rawId(), par[0], par[1], par[2], par[3]);
  } else if (const RectangularPlaneBounds *b2 = dynamic_cast<const RectangularPlaneBounds *>(b)) {
    // Rectangular
    float dx = b2->width() * 0.5;   // half width
    float dy = b2->length() * 0.5;  // half length
    xy[0][0] = -dx;
    xy[0][1] = -dy;
    xy[1][0] = -dx;
    xy[1][1] = dy;
    xy[2][0] = dx;
    xy[2][1] = dy;
    xy[3][0] = dx;
    xy[3][1] = -dy;
    dz = b2->thickness() * 0.5;  // half thickness

    // printf("RECT 0x%x %f %f %f\n", detid.rawId(), dx, dy, dz);
  } else {
    throw cms::Exception("UnimplementedFeature") << "unsupported Bounds class";
  }

  const bool useMatched = false;
  int lay =
      layerNrConv_.convertLayerNumber(detid.subdetId(),
                                      trackerTopo_->layer(detid),
                                      useMatched,
                                      trackerTopo_->isStereo(detid),
                                      trackerTopo_->side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap));

  mkfit::LayerInfo &layer_info = trk_info.layer_nc(lay);
  if (lgc_map) {
    (*lgc_map)[lay].reset_current();
  }
  for (int i = 0; i < 4; ++i) {
    Local3DPoint lp1(xy[i][0], xy[i][1], -dz);
    Local3DPoint lp2(xy[i][0], xy[i][1], dz);
    GlobalPoint gp1 = det->surface().toGlobal(lp1);
    GlobalPoint gp2 = det->surface().toGlobal(lp2);
    considerPoint(gp1, layer_info);
    considerPoint(gp2, layer_info);
    if (lgc_map) {
      (*lgc_map)[lay].extend_current(gp1.perp2());
      (*lgc_map)[lay].extend_current(gp2.perp2());
    }
  }
  if (lgc_map) {
    (*lgc_map)[lay].add_current();
  }
  // Module information
  const auto &p = det->position();
  auto z = det->rotation().z();
  auto x = det->rotation().x();
  layer_info.register_module({{p.x(), p.y(), p.z()}, {z.x(), z.y(), z.z()}, {x.x(), x.y(), x.z()}, detid.rawId()});
  // Set some layer parameters (repeatedly, would require hard-coding otherwise)
  layer_info.set_subdet(detid.subdetId());
  layer_info.set_is_pixel(detid.subdetId() <= 2);
  layer_info.set_is_stereo(trackerTopo_->isStereo(detid));
}

//==============================================================================

// Ideally these functions would also:
// 0. Setup LayerInfo data (which is now done in auto-generated code).
//    Some data-members are a bit over specific, esp/ bools for CMS sub-detectors.
// 1. Establish short module ids (now done in MkFitGeometry constructor).
// 2. Store module normal and strip direction vectors
// 3. ? Any other information ?
// 4. Extract stereo coverage holes where they exist (TEC, all but last 3 double-layers).
//
// Plugin DumpMkFitGeometry.cc can then be used to export this for stand-alone.
// Would also need to be picked up with tk-ntuple converter (to get module ids as
// they will now be used as indices into module info vectors).
//
// An attempt at export cmsRun config is in python/dumpMkFitGeometry.py

void MkFitGeometryESProducer::addPixBGeometry(mkfit::TrackerInfo &trk_info) {
  for (auto &det : trackerGeom_->detsPXB()) {
    fillShapeAndPlacement(det, trk_info);
  }
}

void MkFitGeometryESProducer::addPixEGeometry(mkfit::TrackerInfo &trk_info) {
  for (auto &det : trackerGeom_->detsPXF()) {
    fillShapeAndPlacement(det, trk_info);
  }
}

void MkFitGeometryESProducer::addTIBGeometry(mkfit::TrackerInfo &trk_info) {
  for (auto &det : trackerGeom_->detsTIB()) {
    fillShapeAndPlacement(det, trk_info);
  }
}

void MkFitGeometryESProducer::addTOBGeometry(mkfit::TrackerInfo &trk_info) {
  for (auto &det : trackerGeom_->detsTOB()) {
    fillShapeAndPlacement(det, trk_info);
  }
}

void MkFitGeometryESProducer::addTIDGeometry(mkfit::TrackerInfo &trk_info) {
  for (auto &det : trackerGeom_->detsTID()) {
    fillShapeAndPlacement(det, trk_info);
  }
}

void MkFitGeometryESProducer::addTECGeometry(mkfit::TrackerInfo &trk_info) {
  // For TEC we also need to discover hole in radial extents.
  layer_gap_map_t lgc_map;
  for (auto &det : trackerGeom_->detsTEC()) {
    fillShapeAndPlacement(det, trk_info, &lgc_map);
  }
  // Now loop over the GapCollectors and see if there is a coverage gap.
  std::ostringstream ostr;
  ostr << "addTECGeometry() gap report:\n";
  GapCollector::Interval itvl;
  for (auto &[layer, gcol] : lgc_map) {
    gcol.sqrt_elements();
    if (gcol.find_gap(itvl, 0.5)) {
      ostr << "  layer: " << layer << ", gap: " << itvl.x << " -> " << itvl.y << " width = " << itvl.y - itvl.x << "\n";
      ostr << "    all gaps: ";
      gcol.print_gaps(ostr);
      ostr << "\n";
      trk_info.layer_nc(layer).set_r_hole_range(itvl.x, itvl.y);
    }
  }
  edm::LogVerbatim("MkFitGeometryESProducer") << ostr.str();
}

//------------------------------------------------------------------------------
// clang-format off
namespace {
  const float phase1QBins[] = {
    // PIXB, TIB, TOB
    2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5,
    // PIXE+, TID+, TEC+
    1.0, 1.0, 1.0, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5,
    // PIXE-, TID-, TEC-
    1.0, 1.0, 1.0, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5, 10.25, 7.5
  };
}
// clang-format on
//------------------------------------------------------------------------------

std::unique_ptr<MkFitGeometry> MkFitGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  auto trackerInfo = std::make_unique<mkfit::TrackerInfo>();

  trackerGeom_ = &iRecord.get(geomToken_);
  trackerTopo_ = &iRecord.get(ttopoToken_);

  // std::string path = "Geometry/TrackerCommonData/data/";
  if (trackerGeom_->isThere(GeomDetEnumerators::P1PXB) || trackerGeom_->isThere(GeomDetEnumerators::P1PXEC)) {
    edm::LogInfo("MkFitGeometryESProducer") << "Extracting PhaseI geometry";
    trackerInfo->create_layers(18, 27, 27);
  } else if (trackerGeom_->isThere(GeomDetEnumerators::P2PXB) || trackerGeom_->isThere(GeomDetEnumerators::P2PXEC) ||
             trackerGeom_->isThere(GeomDetEnumerators::P2OTB) || trackerGeom_->isThere(GeomDetEnumerators::P2OTEC)) {
    throw cms::Exception("UnimplementedFeature") << "PhaseII geometry extraction";
  } else {
    throw cms::Exception("UnimplementedFeature") << "unsupported / unknowen geometry version";
  }

  // Prepare layer boundaries for bounding-box search
  for (int i = 0; i < trackerInfo->n_layers(); ++i) {
    auto &li = trackerInfo->layer_nc(i);
    li.set_limits(
        std::numeric_limits<float>::max(), 0, std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    li.reserve_modules(256);
  }
  // This is sort of CMS-2017 specific ... but fireworks code uses it for PhaseII as well.
  addPixBGeometry(*trackerInfo);
  addPixEGeometry(*trackerInfo);
  addTIBGeometry(*trackerInfo);
  addTIDGeometry(*trackerInfo);
  addTOBGeometry(*trackerInfo);
  addTECGeometry(*trackerInfo);

  // r_in/out kept as squres until here, root them
  for (int i = 0; i < trackerInfo->n_layers(); ++i) {
    auto &li = trackerInfo->layer_nc(i);
    li.set_r_in_out(std::sqrt(li.rin()), std::sqrt(li.rout()));
    li.set_propagate_to(li.is_barrel() ? li.r_mean() : li.z_mean());
    li.set_q_bin(phase1QBins[i]);
    unsigned int maxsid = li.shrink_modules();
    // Make sure the short id fits in the 12 bits...
    assert(maxsid < 1u << 11);
  }

  return std::make_unique<MkFitGeometry>(
      iRecord.get(geomToken_), iRecord.get(trackerToken_), iRecord.get(ttopoToken_), std::move(trackerInfo));
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitGeometryESProducer);
