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

#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"

// mkFit includes
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"
#include "RecoTracker/MkFitCore/interface/Config.h"

#include <sstream>

// #define DUMP_MKF_GEO

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

  struct MatHistBin {
    float weight{0}, xi{0}, rl{0};
    void add(float w, float x, float r) {
      weight += w;
      xi += w * x;
      rl += w * r;
    }
  };
  using MaterialHistogram = mkfit::rectvec<MatHistBin>;

  void considerPoint(const GlobalPoint &gp, mkfit::LayerInfo &lay_info);
  void fillShapeAndPlacement(const GeomDet *det,
                             mkfit::TrackerInfo &trk_info,
                             MaterialHistogram &material_histogram,
                             layer_gap_map_t *lgc_map = nullptr);
  void addPixBGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);
  void addPixEGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);
  void addTIBGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);
  void addTOBGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);
  void addTIDGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);
  void addTECGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);

  void findRZBox(const GlobalPoint &gp, float &rmin, float &rmax, float &zmin, float &zmax);
  void aggregateMaterialInfo(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram);
  void fillLayers(mkfit::TrackerInfo &trk_info);

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
                                                    MaterialHistogram &material_histogram,
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

#ifdef DUMP_MKF_GEO
    printf("TRAP 0x%x %f %f %f %f  ", detid.rawId(), par[0], par[1], par[2], par[3]);
#endif
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

#ifdef DUMP_MKF_GEO
    printf("RECT 0x%x %f %f %f  ", detid.rawId(), dx, dy, dz);
#endif
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
#ifdef DUMP_MKF_GEO
  printf("  subdet=%d layer=%d side=%d is_stereo=%d --> mkflayer=%d\n",
         detid.subdetId(),
         trackerTopo_->layer(detid),
         trackerTopo_->side(detid),
         trackerTopo_->isStereo(detid),
         lay);
#endif

  mkfit::LayerInfo &layer_info = trk_info.layer_nc(lay);
  if (lgc_map) {
    (*lgc_map)[lay].reset_current();
  }
  float zbox_min = 1000, zbox_max = 0, rbox_min = 1000, rbox_max = 0;
  for (int i = 0; i < 4; ++i) {
    Local3DPoint lp1(xy[i][0], xy[i][1], -dz);
    Local3DPoint lp2(xy[i][0], xy[i][1], dz);
    GlobalPoint gp1 = det->surface().toGlobal(lp1);
    GlobalPoint gp2 = det->surface().toGlobal(lp2);
    considerPoint(gp1, layer_info);
    considerPoint(gp2, layer_info);
    findRZBox(gp1, rbox_min, rbox_max, zbox_min, zbox_max);
    findRZBox(gp2, rbox_min, rbox_max, zbox_min, zbox_max);
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

  bool doubleSide = false;  //double modules have double material
  if (detid.subdetId() == SiStripSubdetector::TIB)
    doubleSide = trackerTopo_->tibIsDoubleSide(detid);
  else if (detid.subdetId() == SiStripSubdetector::TID)
    doubleSide = trackerTopo_->tidIsDoubleSide(detid);
  else if (detid.subdetId() == SiStripSubdetector::TOB)
    doubleSide = trackerTopo_->tobIsDoubleSide(detid);
  else if (detid.subdetId() == SiStripSubdetector::TEC)
    doubleSide = trackerTopo_->tecIsDoubleSide(detid);

  if (!doubleSide)  //fill material
  {
    //module material
    const float bbxi = det->surface().mediumProperties().xi();
    const float radL = det->surface().mediumProperties().radLen();
    //loop over bins to fill histogram with bbxi, radL and their weight, which the overlap surface in r-z with the cmsquare of a bin
    const float iBin = trk_info.mat_range_z() / trk_info.mat_nbins_z();
    const float jBin = trk_info.mat_range_r() / trk_info.mat_nbins_r();
    for (int i = std::floor(zbox_min / iBin); i < std::ceil(zbox_max / iBin); i++) {
      for (int j = std::floor(rbox_min / jBin); j < std::ceil(rbox_max / jBin); j++) {
        const float iF = i * iBin;
        const float jF = j * jBin;
        float overlap = std::max(0.f, std::min(jF + jBin, rbox_max) - std::max(jF, rbox_min)) *
                        std::max(0.f, std::min(iF + iBin, zbox_max) - std::max(iF, zbox_min));
        if (overlap > 0)
          material_histogram(i, j).add(overlap, bbxi, radL);
      }
    }
  }
}

//==============================================================================

// These functions do the following:
// 0. Detect bounding cylinder of each layer.
// 1. Setup LayerInfo data.
// 2. Establish short module ids.
// 3. Store module normal and strip direction vectors.
// 4. Extract stereo coverage holes where they exist (TEC, all but last 3 double-layers).
//
// See python/dumpMkFitGeometry.py and dumpMkFitGeometryPhase2.py

void MkFitGeometryESProducer::addPixBGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram) {
#ifdef DUMP_MKF_GEO
  printf("\n*** addPixBGeometry\n\n");
#endif
  for (auto &det : trackerGeom_->detsPXB()) {
    fillShapeAndPlacement(det, trk_info, material_histogram);
  }
}

void MkFitGeometryESProducer::addPixEGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram) {
#ifdef DUMP_MKF_GEO
  printf("\n*** addPixEGeometry\n\n");
#endif
  for (auto &det : trackerGeom_->detsPXF()) {
    fillShapeAndPlacement(det, trk_info, material_histogram);
  }
}

void MkFitGeometryESProducer::addTIBGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram) {
#ifdef DUMP_MKF_GEO
  printf("\n*** addTIBGeometry\n\n");
#endif
  for (auto &det : trackerGeom_->detsTIB()) {
    fillShapeAndPlacement(det, trk_info, material_histogram);
  }
}

void MkFitGeometryESProducer::addTOBGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram) {
#ifdef DUMP_MKF_GEO
  printf("\n*** addTOBGeometry\n\n");
#endif
  for (auto &det : trackerGeom_->detsTOB()) {
    fillShapeAndPlacement(det, trk_info, material_histogram);
  }
}

void MkFitGeometryESProducer::addTIDGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram) {
#ifdef DUMP_MKF_GEO
  printf("\n*** addTIDGeometry\n\n");
#endif
  for (auto &det : trackerGeom_->detsTID()) {
    fillShapeAndPlacement(det, trk_info, material_histogram);
  }
}

void MkFitGeometryESProducer::addTECGeometry(mkfit::TrackerInfo &trk_info, MaterialHistogram &material_histogram) {
#ifdef DUMP_MKF_GEO
  printf("\n*** addTECGeometry\n\n");
#endif
  // For TEC we also need to discover hole in radial extents.
  layer_gap_map_t lgc_map;
  for (auto &det : trackerGeom_->detsTEC()) {
    fillShapeAndPlacement(det, trk_info, material_histogram, &lgc_map);
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

void MkFitGeometryESProducer::findRZBox(const GlobalPoint &gp, float &rmin, float &rmax, float &zmin, float &zmax) {
  float r = gp.perp(), z = gp.z();
  if (std::fabs(r) > rmax)
    rmax = std::fabs(r);
  if (std::fabs(r) < rmin)
    rmin = std::fabs(r);
  if (std::fabs(z) > zmax)
    zmax = std::fabs(z);
  if (std::fabs(z) < zmin)
    zmin = std::fabs(z);
}

void MkFitGeometryESProducer::aggregateMaterialInfo(mkfit::TrackerInfo &trk_info,
                                                    MaterialHistogram &material_histogram) {
  //from histogram (vector of tuples) to grid
  for (int i = 0; i < trk_info.mat_nbins_z(); i++) {
    for (int j = 0; j < trk_info.mat_nbins_r(); j++) {
      const MatHistBin &mhb = material_histogram(i, j);
      if (mhb.weight > 0) {
        trk_info.material_bbxi(i, j) = mhb.xi / mhb.weight;
        trk_info.material_radl(i, j) = mhb.rl / mhb.weight;
      }
    }
  }
}

void MkFitGeometryESProducer::fillLayers(mkfit::TrackerInfo &trk_info) {
  mkfit::rectvec<int> rneighbor_map(trk_info.mat_nbins_z(), trk_info.mat_nbins_r());
  mkfit::rectvec<int> zneighbor_map(trk_info.mat_nbins_z(), trk_info.mat_nbins_r());

  for (int im = 0; im < trk_info.n_layers(); ++im) {
    const mkfit::LayerInfo &li = trk_info.layer(im);
    if (!li.is_barrel() && li.zmax() < 0)
      continue;  // neg endcap covered by pos
    int rin, rout, zmin, zmax;
    rin = trk_info.mat_bin_r(li.rin());
    rout = trk_info.mat_bin_r(li.rout()) + 1;
    if (li.is_barrel()) {
      zmin = 0;
      zmax = trk_info.mat_bin_z(std::max(std::abs(li.zmax()), std::abs(li.zmin()))) + 1;
    } else {
      zmin = trk_info.mat_bin_z(li.zmin());
      zmax = trk_info.mat_bin_z(li.zmax()) + 1;
    }
    for (int i = zmin; i < zmax; i++) {
      for (int j = rin; j < rout; j++) {
        if (trk_info.material_bbxi(i, j) == 0) {
          float distancesqmin = 100000;
          for (int i2 = zmin; i2 < zmax; i2++) {
            for (int j2 = rin; j2 < rout; j2++) {
              if (j == j2 && i == i2)
                continue;
              auto mydistsq = (i - i2) * (i - i2) + (j - j2) * (j - j2);
              if (mydistsq < distancesqmin && trk_info.material_radl(i2, j2) > 0) {
                distancesqmin = mydistsq;
                zneighbor_map(i, j) = i2;
                rneighbor_map(i, j) = j2;
              }
            }
          }  // can work on speedup here
        }
      }
    }
    for (int i = zmin; i < zmax; i++) {
      for (int j = rin; j < rout; j++) {
        if (trk_info.material_bbxi(i, j) == 0) {
          int iN = zneighbor_map(i, j);
          int jN = rneighbor_map(i, j);
          trk_info.material_bbxi(i, j) = trk_info.material_bbxi(iN, jN);
          trk_info.material_radl(i, j) = trk_info.material_radl(iN, jN);
        }
      }
    }
  }  //module loop
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
  const float phase2QBins[] = {
    // TODO: Review these numbers.
    // PIXB, TOB
    2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
    // PIXE+, TEC+
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
    // PIXE-, TEC-
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6
  };
}
// clang-format on
//------------------------------------------------------------------------------

std::unique_ptr<MkFitGeometry> MkFitGeometryESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  auto trackerInfo = std::make_unique<mkfit::TrackerInfo>();

  trackerGeom_ = &iRecord.get(geomToken_);
  trackerTopo_ = &iRecord.get(ttopoToken_);

  const float *qBinDefaults = nullptr;

  // std::string path = "Geometry/TrackerCommonData/data/";
  if (trackerGeom_->isThere(GeomDetEnumerators::P1PXB) || trackerGeom_->isThere(GeomDetEnumerators::P1PXEC)) {
    edm::LogInfo("MkFitGeometryESProducer") << "Extracting PhaseI geometry";
    trackerInfo->create_layers(18, 27, 27);
    qBinDefaults = phase1QBins;

    trackerInfo->create_material(300, 300.0f, 120, 120.0f);
  } else if (trackerGeom_->isThere(GeomDetEnumerators::P2PXB) || trackerGeom_->isThere(GeomDetEnumerators::P2PXEC) ||
             trackerGeom_->isThere(GeomDetEnumerators::P2OTB) || trackerGeom_->isThere(GeomDetEnumerators::P2OTEC)) {
    edm::LogInfo("MkFitGeometryESProducer") << "Extracting PhaseII geometry";
    layerNrConv_.reset(mkfit::TkLayout::phase2);
    trackerInfo->create_layers(16, 22, 22);
    qBinDefaults = phase2QBins;
    trackerInfo->create_material(300, 300.0f, 120, 120.0f);
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

  MaterialHistogram material_histogram(trackerInfo->mat_nbins_z(), trackerInfo->mat_nbins_r());

  // This is sort of CMS-phase1 specific ... but fireworks code uses it for PhaseII as well.
  addPixBGeometry(*trackerInfo, material_histogram);
  addPixEGeometry(*trackerInfo, material_histogram);
  addTIBGeometry(*trackerInfo, material_histogram);
  addTIDGeometry(*trackerInfo, material_histogram);
  addTOBGeometry(*trackerInfo, material_histogram);
  addTECGeometry(*trackerInfo, material_histogram);

  // r_in/out kept as squares until here, root them
  unsigned int n_mod = 0;
  for (int i = 0; i < trackerInfo->n_layers(); ++i) {
    auto &li = trackerInfo->layer_nc(i);
    li.set_r_in_out(std::sqrt(li.rin()), std::sqrt(li.rout()));
    li.set_propagate_to(li.is_barrel() ? li.r_mean() : li.z_mean());
    li.set_q_bin(qBinDefaults[i]);
    unsigned int maxsid = li.shrink_modules();

    n_mod += maxsid;

    // Make sure the short id fits in the 14 bits...
    assert(maxsid < 1u << 13);
    assert(n_mod > 0);
  }

  // Material grid
  aggregateMaterialInfo(*trackerInfo, material_histogram);
  fillLayers(*trackerInfo);

  // Propagation configuration
  {
    using namespace mkfit;
    PropagationConfig &pconf = trackerInfo->prop_config_nc();
    pconf.backward_fit_to_pca = false;
    pconf.finding_requires_propagation_to_hit_pos = true;
    pconf.finding_inter_layer_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    pconf.finding_intra_layer_pflags = PropagationFlags(PF_none);
    pconf.backward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    pconf.forward_fit_pflags = PropagationFlags(PF_use_param_b_field | PF_apply_material);
    pconf.seed_fit_pflags = PropagationFlags(PF_none);
    pconf.pca_prop_pflags = PropagationFlags(PF_none);
    pconf.apply_tracker_info(trackerInfo.get());
  }

#ifdef DUMP_MKF_GEO
  printf("Total number of modules %u, 14-bits fit up to %u modules\n", n_mod, 1u << 13);
#endif

  return std::make_unique<MkFitGeometry>(iRecord.get(geomToken_),
                                         iRecord.get(trackerToken_),
                                         iRecord.get(ttopoToken_),
                                         std::move(trackerInfo),
                                         layerNrConv_);
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitGeometryESProducer);
