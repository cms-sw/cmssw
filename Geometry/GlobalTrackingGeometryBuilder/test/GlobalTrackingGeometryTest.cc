/*
 * \file GlobalTrackingGeometryTest.cc
 *
 *  $Date: 2011/09/27 12:06:32 $
 *  $Revision: 1.5 $
 *  \author M. Sani
 */

#include <memory>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <Geometry/GlobalTrackingGeometryBuilder/test/GlobalTrackingGeometryTest.h>
#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>

#include <DataFormats/DetId/interface/DetId.h>

#include <string>
#include <cmath>
#include <iomanip> 
#include <vector>

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
  const TrackerGeometry* trackerGeometry = 0;
  std::cout << "Pointer to Tracker Geometry: ";
  try {
    trackerGeometry = (const TrackerGeometry*) geo->slaveGeometry(detId1);
    std::cout << trackerGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
    
  DetId detId2(DetId::Muon, 1); 
  const DTGeometry* dtGeometry = 0;
  std::cout << "Pointer to DT Geometry: ";
  try {
    dtGeometry = (const DTGeometry*) geo->slaveGeometry(detId2);
    std::cout << dtGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
 
  DetId detId3(DetId::Muon, 2); 
  const CSCGeometry* cscGeometry = 0;
  std::cout << "Pointer to CSC Geometry: "; 
  try {
    cscGeometry = (const CSCGeometry*) geo->slaveGeometry(detId3);
    std::cout << cscGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
 
  DetId detId4(DetId::Muon, 3); 
  const RPCGeometry* rpcGeometry = 0;
  std::cout << "Pointer to RPC Geometry: ";
  try {
    rpcGeometry = (const RPCGeometry*) geo->slaveGeometry(detId4);
    std::cout <<  rpcGeometry << std::endl;
  } catch (...) {
    std::cout << "N/A" << std::endl;
  }
    
  DetId detId5(DetId::Muon, 4); 
  const GEMGeometry* gemGeometry = 0;
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
  if (trackerGeometry) analyzeTracker(geo.product(), trackerGeometry);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalTrackingGeometryTest);
