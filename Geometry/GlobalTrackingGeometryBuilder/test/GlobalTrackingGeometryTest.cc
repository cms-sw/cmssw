/*
 * \file GlobalTrackingGeometryTest.cc
 *
 *  \author M. Sani
 */

#include <memory>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <string>
#include <cmath>
#include <iomanip>
#include <vector>

class GlobalTrackingGeometryTest : public edm::one::EDAnalyzer<> {
public:
  explicit GlobalTrackingGeometryTest(const edm::ParameterSet&);
  ~GlobalTrackingGeometryTest() override = default;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void analyzeCSC(const GlobalTrackingGeometry* geo, const CSCGeometry* cscGeometry);
  void analyzeDT(const GlobalTrackingGeometry* geo, const DTGeometry* dtGeometry);
  void analyzeRPC(const GlobalTrackingGeometry* geo, const RPCGeometry* rpcGeometry);
  void analyzeGEM(const GlobalTrackingGeometry* geo, const GEMGeometry* gemGeometry);
  void analyzeMTD(const GlobalTrackingGeometry* geo, const MTDGeometry* mtdGeometry);
  void analyzeTracker(const GlobalTrackingGeometry* geo, const TrackerGeometry* tkGeometry);
  const std::string& myName() { return my_name; }
  std::string my_name;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;
};

GlobalTrackingGeometryTest::GlobalTrackingGeometryTest(const edm::ParameterSet& /*iConfig*/)
    : my_name("GlobalTrackingGeometryTest"),
      geometryToken_{esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>(edm::ESInputTag{})} {}

void GlobalTrackingGeometryTest::analyzeCSC(const GlobalTrackingGeometry* geo, const CSCGeometry* cscGeometry) {
  for (auto& detUnit : cscGeometry->detUnits()) {
    const DetId detId(detUnit->geographicalId());

    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  edm::LogVerbatim("GlobalTracking") << "CSC detUnit: GlobalTrackingGeometry succesfully tested.";

  for (auto& det : cscGeometry->dets()) {
    const DetId detId(det->geographicalId());

    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  edm::LogVerbatim("GlobalTracking") << "CSC det: GlobalTrackingGeometry succesfully tested.";
}

void GlobalTrackingGeometryTest::analyzeDT(const GlobalTrackingGeometry* geo, const DTGeometry* dtGeometry) {
  for (auto& detUnit : dtGeometry->detUnits()) {
    const DetId detId(detUnit->geographicalId());

    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  edm::LogVerbatim("GlobalTracking") << "DT detUnit: GlobalTrackingGeometry succesfully tested.";

  for (auto& det : dtGeometry->dets()) {
    const DetId detId(det->geographicalId());

    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  edm::LogVerbatim("GlobalTracking") << "DT det: GlobalTrackingGeometry succesfully tested.";
}

void GlobalTrackingGeometryTest::analyzeRPC(const GlobalTrackingGeometry* geo, const RPCGeometry* rpcGeometry) {
  for (auto& detUnit : rpcGeometry->detUnits()) {
    const DetId detId(detUnit->geographicalId());

    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  edm::LogVerbatim("GlobalTracking") << "RPC detUnit: GlobalTrackingGeometry succesfully tested.";

  for (auto& det : rpcGeometry->dets()) {
    const DetId detId(det->geographicalId());

    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  edm::LogVerbatim("GlobalTracking") << "RPC det: GlobalTrackingGeometry succesfully tested.";
}

void GlobalTrackingGeometryTest::analyzeGEM(const GlobalTrackingGeometry* geo, const GEMGeometry* gemGeometry) {
  for (auto& detUnit : gemGeometry->detUnits()) {
    const DetId detId(detUnit->geographicalId());

    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  edm::LogVerbatim("GlobalTracking") << "GEM detUnit: GlobalTrackingGeometry succesfully tested.";

  for (auto& det : gemGeometry->dets()) {
    const DetId detId(det->geographicalId());

    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  edm::LogVerbatim("GlobalTracking") << "GEM det: GlobalTrackingGeometry succesfully tested.";
}

void GlobalTrackingGeometryTest::analyzeMTD(const GlobalTrackingGeometry* geo, const MTDGeometry* mtdGeometry) {
  for (auto& detUnit : mtdGeometry->detUnits()) {
    const DetId detId(detUnit->geographicalId());

    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  edm::LogVerbatim("GlobalTracking") << "MTD detUnit: GlobalTrackingGeometry succesfully tested.";

  for (auto& det : mtdGeometry->dets()) {
    const DetId detId(det->geographicalId());

    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  edm::LogVerbatim("GlobalTracking") << "MTD det: GlobalTrackingGeometry succesfully tested.";
}

void GlobalTrackingGeometryTest::analyzeTracker(const GlobalTrackingGeometry* geo, const TrackerGeometry* tkGeometry) {
  for (auto& detUnit : tkGeometry->detUnits()) {
    const DetId detId(detUnit->geographicalId());

    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  edm::LogVerbatim("GlobalTracking") << "Tracker detUnit: GlobalTrackingGeometry succesfully tested.";

  for (auto& det : tkGeometry->dets()) {
    const DetId detId(det->geographicalId());

    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  edm::LogVerbatim("GlobalTracking") << "Tracker det: GlobalTrackingGeometry succesfully tested.";
}

void GlobalTrackingGeometryTest::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("GlobalTracking") << myName() << ": Analyzer...";

  const auto& geo = iSetup.getData(geometryToken_);

  DetId detId1(DetId::Tracker, 0);
  const TrackerGeometry* trackerGeometry = nullptr;
  try {
    trackerGeometry = (const TrackerGeometry*)geo.slaveGeometry(detId1);
    edm::LogVerbatim("GlobalTracking") << "Pointer to Tracker Geometry: " << trackerGeometry;
  } catch (...) {
    edm::LogVerbatim("GlobalTracking") << "Pointer to Tracker Geometry: N/A";
  }

  DetId detId6(DetId::Forward, 1);
  const MTDGeometry* mtdGeometry = nullptr;
  try {
    mtdGeometry = (const MTDGeometry*)geo.slaveGeometry(detId6);
    edm::LogVerbatim("GlobalTracking") << "Pointer to MTD Geometry: " << mtdGeometry;
  } catch (...) {
    edm::LogVerbatim("GlobalTracking") << "Pointer to MTD Geometry: N/A";
  }

  DetId detId2(DetId::Muon, 1);
  const DTGeometry* dtGeometry = nullptr;
  try {
    dtGeometry = (const DTGeometry*)geo.slaveGeometry(detId2);
    edm::LogVerbatim("GlobalTracking") << "Pointer to DT Geometry: " << dtGeometry;
  } catch (...) {
    edm::LogVerbatim("GlobalTracking") << "Pointer to DT Geometry: N/A";
  }

  DetId detId3(DetId::Muon, 2);
  const CSCGeometry* cscGeometry = nullptr;
  try {
    cscGeometry = (const CSCGeometry*)geo.slaveGeometry(detId3);
    edm::LogVerbatim("GlobalTracking") << "Pointer to CSC Geometry: " << cscGeometry;
  } catch (...) {
    edm::LogVerbatim("GlobalTracking") << "Pointer to CSC Geometry: N/A";
  }

  DetId detId4(DetId::Muon, 3);
  const RPCGeometry* rpcGeometry = nullptr;
  try {
    rpcGeometry = (const RPCGeometry*)geo.slaveGeometry(detId4);
    edm::LogVerbatim("GlobalTracking") << "Pointer to RPC Geometry: " << rpcGeometry;
  } catch (...) {
    edm::LogVerbatim("GlobalTracking") << "Pointer to RPC Geometry: N/A";
  }

  DetId detId5(DetId::Muon, 4);
  const GEMGeometry* gemGeometry = nullptr;
  try {
    gemGeometry = (const GEMGeometry*)geo.slaveGeometry(detId5);
    edm::LogVerbatim("GlobalTracking") << "Pointer to GEM Geometry: " << gemGeometry;
  } catch (...) {
    edm::LogVerbatim("GlobalTracking") << "Pointer to GEM Geometry: N/A";
  }

  if (cscGeometry)
    analyzeCSC(&geo, cscGeometry);
  if (dtGeometry)
    analyzeDT(&geo, dtGeometry);
  if (rpcGeometry)
    analyzeRPC(&geo, rpcGeometry);
  if (gemGeometry)
    analyzeGEM(&geo, gemGeometry);
  if (mtdGeometry)
    analyzeMTD(&geo, mtdGeometry);
  if (trackerGeometry)
    analyzeTracker(&geo, trackerGeometry);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalTrackingGeometryTest);
