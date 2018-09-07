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

class GlobalTrackingGeometryTest : public edm::one::EDAnalyzer<>
{
public:
 
  explicit GlobalTrackingGeometryTest( const edm::ParameterSet& );
  ~GlobalTrackingGeometryTest() override;

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
};

GlobalTrackingGeometryTest::GlobalTrackingGeometryTest( const edm::ParameterSet& /*iConfig*/)
 : my_name( "GlobalTrackingGeometryTest" ) {}

GlobalTrackingGeometryTest::~GlobalTrackingGeometryTest() {}

void GlobalTrackingGeometryTest::analyzeCSC(const GlobalTrackingGeometry* geo, const CSCGeometry* cscGeometry) 
{
  for(auto& detUnit : cscGeometry->detUnits()) 
  {
    const DetId detId(detUnit->geographicalId());
        
    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  std::cout << "CSC detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
  for(auto& det : cscGeometry->dets()) 
  {
    const DetId detId(det->geographicalId());
        
    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  std::cout << "CSC det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeDT(const GlobalTrackingGeometry* geo, const DTGeometry* dtGeometry) 
{
  for(auto& detUnit : dtGeometry->detUnits()) 
  {
    const DetId detId(detUnit->geographicalId());
        
    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  std::cout << "DT detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
  for(auto& det : dtGeometry->dets()) 
  {
    const DetId detId(det->geographicalId());
        
    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  std::cout << "DT det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeRPC(const GlobalTrackingGeometry* geo, const RPCGeometry* rpcGeometry) 
{
  for(auto& detUnit : rpcGeometry->detUnits()) 
  {
    const DetId detId(detUnit->geographicalId());
        
    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  std::cout << "RPC detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
  for(auto& det : rpcGeometry->dets()) 
  {
    const DetId detId(det->geographicalId());
        
    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  std::cout << "RPC det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeGEM(const GlobalTrackingGeometry* geo, const GEMGeometry* gemGeometry) 
{
  for(auto& detUnit : gemGeometry->detUnits()) 
  {
    const DetId detId(detUnit->geographicalId());
        
    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  std::cout << "GEM detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
  for(auto& det : gemGeometry->dets()) 
  {
    const DetId detId(det->geographicalId());
        
    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  std::cout << "GEM det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeMTD(const GlobalTrackingGeometry* geo, const MTDGeometry* mtdGeometry) 
{
  for(auto& detUnit : mtdGeometry->detUnits()) 
  {
    const DetId detId(detUnit->geographicalId());
        
    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  std::cout << "MTD detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
  for(auto& det : mtdGeometry->dets()) 
  {
    const DetId detId(det->geographicalId());
        
    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  std::cout << "MTD det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeTracker(const GlobalTrackingGeometry* geo, const TrackerGeometry* tkGeometry) 
{
  for(auto& detUnit : tkGeometry->detUnits()) 
  {
    const DetId detId(detUnit->geographicalId());
        
    // Check idToDetUnit
    const GeomDetUnit* gdu(geo->idToDetUnit(detId));
    assert(gdu == detUnit);
  }
  std::cout << "Tracker detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
  for(auto& det : tkGeometry->dets()) 
  {
    const DetId detId(det->geographicalId());
        
    // Check idToDet
    const GeomDet* gd(geo->idToDet(detId));
    assert(gd == det);
  }
  std::cout << "Tracker det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  std::cout << myName() << ": Analyzer..." << std::endl;

  edm::ESHandle<GlobalTrackingGeometry> geo;
  iSetup.get<GlobalTrackingGeometryRecord>().get(geo);     
    
  DetId detId1(DetId::Tracker, 0);
  const TrackerGeometry* trackerGeometry = nullptr;
  std::cout << "Pointer to Tracker Geometry: ";
  try {
    trackerGeometry = (const TrackerGeometry*) geo->slaveGeometry(detId1);
    std::cout << trackerGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }

  DetId detId6(DetId::Forward, 1);
  const MTDGeometry* mtdGeometry = nullptr;
  std::cout << "Pointer to MTD Geometry: ";
  try {
    mtdGeometry = (const MTDGeometry*) geo->slaveGeometry(detId6);
    std::cout << mtdGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
    
  DetId detId2(DetId::Muon, 1); 
  const DTGeometry* dtGeometry = nullptr;
  std::cout << "Pointer to DT Geometry: ";
  try {
    dtGeometry = (const DTGeometry*) geo->slaveGeometry(detId2);
    std::cout << dtGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
 
  DetId detId3(DetId::Muon, 2); 
  const CSCGeometry* cscGeometry = nullptr;
  std::cout << "Pointer to CSC Geometry: "; 
  try {
    cscGeometry = (const CSCGeometry*) geo->slaveGeometry(detId3);
    std::cout << cscGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
 
  DetId detId4(DetId::Muon, 3); 
  const RPCGeometry* rpcGeometry = nullptr;
  std::cout << "Pointer to RPC Geometry: ";
  try {
    rpcGeometry = (const RPCGeometry*) geo->slaveGeometry(detId4);
    std::cout <<  rpcGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
    
  DetId detId5(DetId::Muon, 4); 
  const GEMGeometry* gemGeometry = nullptr;
  std::cout << "Pointer to GEM Geometry: ";
  try {
    gemGeometry = (const GEMGeometry*) geo->slaveGeometry(detId5);
    std::cout <<  gemGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }

  if (cscGeometry) analyzeCSC(geo.product(), cscGeometry);
  if (dtGeometry) analyzeDT(geo.product(), dtGeometry);
  if (rpcGeometry) analyzeRPC(geo.product(), rpcGeometry);
  if (gemGeometry) analyzeGEM(geo.product(), gemGeometry);
  if (mtdGeometry) analyzeMTD(geo.product(), mtdGeometry);
  if (trackerGeometry) analyzeTracker(geo.product(), trackerGeometry);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalTrackingGeometryTest);
