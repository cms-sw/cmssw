// Validation and benchmark for HGCalFastGeometry.
//
// Round trip (always): for valid DetIds of the loaded geometry, sampled over the
// whole detector, take HGCalGeometry::getPosition(id) and push it back through
// HGCalFastGeometry::toDetId(). The recovered DetId must equal the original.
// This is the same contract Geometry/HGCalGeometry/test/HGCalGeomLocatorTester
// checks for getClosestCell(), so the two are directly comparable -- including
// the fact that getClosestCell() itself is not perfectly invertible (see the
// known-mismatch lists in Geometry/HGCalGeometry/data/).
//
// Every disagreement is arbitrated with getClosestCell() on the same point. If
// that does not return the original DetId either, the cell does not survive the
// round trip in HGCalGeometry itself (e.g. negative-z fine-granularity
// scintillator tiles of the v19 geometries, which come back one ring off); such
// cells are reported but not counted. Any other disagreement makes the module
// throw, so it can serve as a unit test.
//
// Benchmark (optional): time toDetId() against HGCalGeometry::getClosestCell()
// on the same set of points.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "FastSimulation/CaloGeometryTools/interface/HGCalFastGeometry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <tuple>
#include <string>
#include <vector>

namespace {
  DetId::Detector detectorOf(const std::string& nameSense) {
    if (nameSense == "HGCalEESensitive")
      return DetId::HGCalEE;
    if (nameSense == "HGCalHESiliconSensitive")
      return DetId::HGCalHSi;
    if (nameSense == "HGCalHEScintillatorSensitive")
      return DetId::HGCalHSc;
    throw cms::Exception("Configuration") << "HGCalFastGeometryTester: unknown nameSense '" << nameSense << "'";
  }
}  // namespace

class HGCalFastGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalFastGeometryTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const std::string nameSense_;
  const DetId::Detector det_;
  const bool benchmark_;
  const int maxCells_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalFastGeometryTester::HGCalFastGeometryTester(const edm::ParameterSet& ps)
    : nameSense_(ps.getParameter<std::string>("nameSense")),
      det_(detectorOf(nameSense_)),
      benchmark_(ps.getParameter<bool>("benchmark")),
      maxCells_(ps.getParameter<int>("maxCells")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {}

void HGCalFastGeometryTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<bool>("benchmark", true);
  desc.add<int>("maxCells", -1);  // -1 = all, otherwise sampled uniformly over the valid DetIds
  descriptions.add("hgcalFastGeometryTester", desc);
}

void HGCalFastGeometryTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& ddd = iSetup.getData(dddToken_);
  const HGCalGeometry& geom = iSetup.getData(geomToken_);

  auto t0 = std::chrono::steady_clock::now();
  HGCalFastGeometry fast(ddd, det_);
  auto t1 = std::chrono::steady_clock::now();
  const double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  edm::LogVerbatim("HGCalFastGeom") << "[" << nameSense_ << "] built in " << buildMs << " ms; layers "
                                    << fast.firstLayer() << ".." << fast.lastLayer() << " (" << fast.nLayers()
                                    << "), cached wafers " << fast.nCachedWafers();

  const std::vector<DetId>& ids = geom.getValidDetIds();
  const std::size_t step = (maxCells_ > 0 && ids.size() > static_cast<std::size_t>(maxCells_))
                               ? (ids.size() + maxCells_ - 1) / maxCells_
                               : 1;
  edm::LogVerbatim("HGCalFastGeom") << "valid DetIds: " << ids.size() << ", testing every " << step;

  unsigned nTested = 0, nNull = 0, nMismatch = 0, nGeomNull = 0, nGeomMismatch = 0;
  std::map<int, unsigned> mismatchByLayer;
  std::map<std::tuple<int, int, int, int>, unsigned> mismatchScint;
  unsigned nRefAgreesFast = 0;
  std::vector<GlobalPoint> points;
  points.reserve(ids.size() / step + 1);

  for (std::size_t k = 0; k < ids.size(); k += step) {
    const DetId id = ids[k];
    const GlobalPoint gp = geom.getPosition(id);
    points.push_back(gp);

    const bool scint = (id.det() == DetId::HGCalHSc);
    const int layer = scint ? HGCScintillatorDetId(id).layer() : HGCSiliconDetId(id).layer();
    const int zside = scint ? HGCScintillatorDetId(id).zside() : HGCSiliconDetId(id).zside();

    const DetId back = fast.toDetId(layer, gp.x(), gp.y(), zside);
    ++nTested;
    if (back.rawId() == id.rawId())
      continue;

    // Arbitrate with the CMSSW lookup on the same point: only a cell that
    // getClosestCell() maps back to itself is a failure of the fast lookup.
    const DetId ref = geom.getClosestCell(gp);
    const bool geomInvertible = (ref.rawId() == id.rawId());
    const bool isNull = (back == DetId());
    if (geomInvertible && isNull)
      ++nNull;
    else if (geomInvertible)
      ++nMismatch;
    else if (isNull)
      ++nGeomNull;
    else
      ++nGeomMismatch;
    if (!geomInvertible && ref.rawId() == back.rawId())
      ++nRefAgreesFast;
    ++mismatchByLayer[layer];
    if (scint && !isNull) {
      const HGCScintillatorDetId e(id), g(back);
      ++mismatchScint[std::make_tuple(e.zside(), int(e.granularity()), g.ring() - e.ring(), g.iphi() - e.iphi())];
    }

    const unsigned nShown = nNull + nMismatch + nGeomNull + nGeomMismatch;
    if (nShown > 10)
      continue;
    const char* tag = geomInvertible ? "  FAIL" : "  not invertible in HGCalGeometry either";
    if (isNull) {
      edm::LogVerbatim("HGCalFastGeom") << tag << ": null DetId at (" << gp.x() << ", " << gp.y() << ", " << gp.z()
                                        << ") for raw id 0x" << std::hex << id.rawId() << std::dec;
    } else if (scint) {
      edm::LogVerbatim("HGCalFastGeom") << tag << ": at (" << gp.x() << ", " << gp.y() << ", " << gp.z() << ")"
                                        << " expected " << HGCScintillatorDetId(id) << " | got "
                                        << HGCScintillatorDetId(back);
    } else {
      HGCSiliconDetId sid(id);
      HGCSiliconDetId b(back);
      edm::LogVerbatim("HGCalFastGeom") << tag << ": at (" << gp.x() << ", " << gp.y() << ", " << gp.z() << ")"
                                        << " expected L" << layer << " w(" << sid.waferU() << "," << sid.waferV()
                                        << ") c(" << sid.cellU() << "," << sid.cellV() << ") t" << sid.type()
                                        << " | got L" << b.layer() << " w(" << b.waferU() << "," << b.waferV() << ") c("
                                        << b.cellU() << "," << b.cellV() << ") t" << b.type();
    }
  }

  const unsigned nFail = nNull + nMismatch;
  const unsigned nGeom = nGeomNull + nGeomMismatch;
  const double failPct = nTested ? 100. * nFail / nTested : 0.;
  const double geomPct = nTested ? 100. * nGeom / nTested : 0.;
  edm::LogVerbatim("HGCalFastGeom") << "RESULT [" << nameSense_ << "] tested " << nTested << "  failed " << nFail
                                    << " (null " << nNull << ", mismatched " << nMismatch << ") -> " << failPct
                                    << "%;  not invertible in HGCalGeometry either " << nGeom << " (null " << nGeomNull
                                    << ", mismatched " << nGeomMismatch << ") -> " << geomPct << "%";
  for (const auto& m : mismatchByLayer)
    edm::LogVerbatim("HGCalFastGeom") << "   layer " << m.first << ": " << m.second;
  if (nGeom > 0)
    edm::LogVerbatim("HGCalFastGeom") << "   on cells HGCalGeometry cannot invert, getClosestCell agrees with the "
                                         "fast lookup "
                                      << nRefAgreesFast << " times";
  for (const auto& m : mismatchScint)
    edm::LogVerbatim("HGCalFastGeom") << "   scint zside " << std::get<0>(m.first) << " granularity "
                                      << std::get<1>(m.first) << " dring " << std::get<2>(m.first) << " dphi "
                                      << std::get<3>(m.first) << ": " << m.second;

  if (nFail > 0)
    throw cms::Exception("HGCalFastGeometryTester")
        << "[" << nameSense_ << "] " << nFail << " of " << nTested
        << " cells are mapped back to themselves by HGCalGeometry::getClosestCell() but not by "
           "HGCalFastGeometry::toDetId()";

  if (benchmark_ && !points.empty()) {
    // toDetId
    auto b0 = std::chrono::steady_clock::now();
    unsigned long long sink = 0;
    for (const auto& gp : points) {
      const int layer = ddd.getLayer(std::abs(gp.z()), true);
      sink += fast.toDetId(layer, gp.x(), gp.y(), (gp.z() > 0 ? 1 : -1)).rawId();
    }
    auto b1 = std::chrono::steady_clock::now();

    // getClosestCell, on a subset -- it is orders of magnitude slower
    const std::size_t nSlow = std::min<std::size_t>(points.size(), 2000);
    auto c0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < nSlow; ++i)
      sink += geom.getClosestCell(points[i]).rawId();
    auto c1 = std::chrono::steady_clock::now();

    const double fastUs = std::chrono::duration<double, std::micro>(b1 - b0).count() / points.size();
    const double slowUs = std::chrono::duration<double, std::micro>(c1 - c0).count() / nSlow;
    edm::LogVerbatim("HGCalFastGeom") << "BENCH [" << nameSense_ << "] toDetId " << fastUs << " us/call ("
                                      << points.size() << " calls); getClosestCell " << slowUs << " us/call (" << nSlow
                                      << " calls); speedup x" << (slowUs / fastUs) << " [sink " << sink << "]";
  }
}

DEFINE_FWK_MODULE(HGCalFastGeometryTester);
