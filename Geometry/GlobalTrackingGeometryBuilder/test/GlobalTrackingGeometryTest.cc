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
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>

#include <DataFormats/DetId/interface/DetId.h>

#include <string>
#include <cmath>
#include <iomanip> 
#include <vector>

GlobalTrackingGeometryTest::GlobalTrackingGeometryTest( const edm::ParameterSet& /*iConfig*/)
 : my_name( "GlobalTrackingGeometryTest" ) {}

GlobalTrackingGeometryTest::~GlobalTrackingGeometryTest() {}

void GlobalTrackingGeometryTest::analyzeCSC(const GlobalTrackingGeometry* geo, const CSCGeometry* cscGeometry) {

    for(CSCGeometry::DetUnitContainer::const_iterator itCSC = cscGeometry->detUnits().begin(); 
        itCSC != cscGeometry->detUnits().end(); itCSC++) {
        
        DetId detId = (*itCSC)->geographicalId();
        
        // Check idToDetUnit
        const GeomDetUnit* gdu = geo->idToDetUnit(detId);
	    assert(gdu == (*itCSC));
    }
    std::cout << "CSC detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
    for(CSCGeometry::DetContainer::const_iterator itCSC = cscGeometry->dets().begin(); 
        itCSC != cscGeometry->dets().end(); itCSC++) {
    
        DetId detId = (*itCSC)->geographicalId();
        
	    // Check idToDet
	    const GeomDet* gd = geo->idToDet(detId);
	    assert(gd == (*itCSC));
    }
    std::cout << "CSC det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeDT(const GlobalTrackingGeometry* geo, const DTGeometry* dtGeometry) 
{
    for(DTGeometry::DetUnitContainer::const_iterator itDT = dtGeometry->detUnits().begin(); 
        itDT != dtGeometry->detUnits().end(); itDT++) {
        
        DetId detId = (*itDT)->geographicalId();
        
        // Check idToDetUnit
        const GeomDetUnit* gdu = geo->idToDetUnit(detId);
	    assert(gdu == (*itDT));
    }
    std::cout << "DT detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
    for(DTGeometry::DetContainer::const_iterator itDT = dtGeometry->dets().begin(); 
        itDT != dtGeometry->dets().end(); itDT++) {
    
        DetId detId = (*itDT)->geographicalId();
        
	    // Check idToDet
	    const GeomDet* gd = geo->idToDet(detId);
	    assert(gd == (*itDT));
    }
    std::cout << "DT det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeRPC(const GlobalTrackingGeometry* geo, const RPCGeometry* rpcGeometry) {

    for(RPCGeometry::DetUnitContainer::const_iterator itRPC = rpcGeometry->detUnits().begin(); 
        itRPC != rpcGeometry->detUnits().end(); itRPC++) {
        
        DetId detId = (*itRPC)->geographicalId();
        
        // Check idToDetUnit
        const GeomDetUnit* gdu = geo->idToDetUnit(detId);
	    assert(gdu == (*itRPC));
    }
    std::cout << "RPC detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
    for(RPCGeometry::DetContainer::const_iterator itRPC = rpcGeometry->dets().begin(); 
        itRPC != rpcGeometry->dets().end(); itRPC++) {
    
        DetId detId = (*itRPC)->geographicalId();

	    // Check idToDet
	    const GeomDet* gd = geo->idToDet(detId);
        assert(gd == (*itRPC));
    }
    std::cout << "RPC det: GlobalTrackingGeometry succesfully tested." << std::endl;
}

void GlobalTrackingGeometryTest::analyzeTracker(const GlobalTrackingGeometry* geo, const TrackerGeometry* tkGeometry) {

    for(TrackerGeometry::DetUnitContainer::const_iterator itTk = tkGeometry->detUnits().begin(); 
        itTk != tkGeometry->detUnits().end(); itTk++) {
        
        DetId detId = (*itTk)->geographicalId();
        
        // Check idToDetUnit
        const GeomDetUnit* gdu = geo->idToDetUnit(detId);
	    assert(gdu == (*itTk));
    }
    std::cout << "Tracker detUnit: GlobalTrackingGeometry succesfully tested." << std::endl;
    
    for(TrackerGeometry::DetContainer::const_iterator itTk = tkGeometry->dets().begin(); 
        itTk != tkGeometry->dets().end(); itTk++) {
    
        DetId detId = (*itTk)->geographicalId();
        
	    // Check idToDet
	    const GeomDet* gd = geo->idToDet(detId);
	    assert(gd == (*itTk));
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
        
    if (cscGeometry) analyzeCSC(geo.product(), cscGeometry);
    if (dtGeometry) analyzeDT(geo.product(), dtGeometry);
    if (rpcGeometry) analyzeRPC(geo.product(), rpcGeometry);
    if (trackerGeometry) analyzeTracker(geo.product(), trackerGeometry);    
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalTrackingGeometryTest);
