/** \file
 *
 *  $Date: 2006/05/03 15:23:17 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/src/MuonDetLayerGeometryESProducer.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>

#include <RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h>
#include <RecoMuon/DetLayers/src/MuonRPCDetLayerGeometryBuilder.h>
#include <RecoMuon/DetLayers/src/MuonDTDetLayerGeometryBuilder.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <memory>

using namespace edm;

MuonDetLayerGeometryESProducer::MuonDetLayerGeometryESProducer(const edm::ParameterSet & p){
  setWhatProduced(this);
}


MuonDetLayerGeometryESProducer::~MuonDetLayerGeometryESProducer(){}


boost::shared_ptr<MuonDetLayerGeometry>
MuonDetLayerGeometryESProducer::produce(const MuonRecoGeometryRecord & record) {

  MuonDetLayerGeometry* muonDetLayerGeometry = new MuonDetLayerGeometry();

  // Build DT layers  
  try {
    edm::ESHandle<DTGeometry> dt;
    record.getRecord<MuonGeometryRecord>().get(dt);
    if (dt.isValid()) {
        muonDetLayerGeometry->addDTLayers(MuonDTDetLayerGeometryBuilder::buildLayers(*dt));
    }
  
  } catch (...) {
    // No DT geo available
    LogInfo("RecoMuonDetLayers") << "No DT geometry is available.";
  }  

  // Build CSC layers
  try {
    edm::ESHandle<CSCGeometry> csc;
    record.getRecord<MuonGeometryRecord>().get(csc);
    if (csc.isValid()) {
        muonDetLayerGeometry->addCSCLayers(MuonCSCDetLayerGeometryBuilder::buildLayers(*csc));
    }
  } catch(...) {
    // No CSC geo available
    LogInfo("RecoMuonDetLayers") << "No CSC geometry is available.";
  }
  
  // Build RPC layers
  try {
    edm::ESHandle<RPCGeometry> rpc;
    record.getRecord<MuonGeometryRecord>().get(rpc);
    if (rpc.isValid()) {
        //muonDetLayerGeometry->addRPCLayers(MuonRPCDetLayerGeometryBuilder::buildBarrelLayers(*rpc));
        //muonDetLayerGeometry->addRPCLayers(MuonRPCDetLayerGeometryBuilder::buildEndcaplLayers(*rpc));
    }
  
  } catch (...) {
    // No RPC geo available
    LogInfo("RecoMuonDetLayers") << "No RPC geometry is available.";
  }  

  return boost::shared_ptr<MuonDetLayerGeometry>(muonDetLayerGeometry);
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonDetLayerGeometryESProducer)
