/** \file
 *
 *  \author N. Amapane - CERN
 *
 *  \modified by R. Radogna & C. Calabria & A. Sharma
 */

#include <RecoMuon/DetLayers/plugins/MuonDetLayerGeometryESProducer.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>

#include <RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h>
#include <RecoMuon/DetLayers/src/MuonRPCDetLayerGeometryBuilder.h>
#include <RecoMuon/DetLayers/src/MuonDTDetLayerGeometryBuilder.h>
#include <RecoMuon/DetLayers/src/MuonGEMDetLayerGeometryBuilder.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Framework/interface/NoProxyException.h>

#include <memory>

using namespace edm;

MuonDetLayerGeometryESProducer::MuonDetLayerGeometryESProducer(const edm::ParameterSet & p){
  setWhatProduced(this);
}


MuonDetLayerGeometryESProducer::~MuonDetLayerGeometryESProducer(){}


boost::shared_ptr<MuonDetLayerGeometry>
MuonDetLayerGeometryESProducer::produce(const MuonRecoGeometryRecord & record) {

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonDetLayerGeometryESProducer";
  MuonDetLayerGeometry* muonDetLayerGeometry = new MuonDetLayerGeometry();
  
  // Build DT layers  
  edm::ESHandle<DTGeometry> dt;
  record.getRecord<MuonGeometryRecord>().get(dt);
  if (dt.isValid()) {
    muonDetLayerGeometry->addDTLayers(MuonDTDetLayerGeometryBuilder::buildLayers(*dt));
  } else {
    LogInfo(metname) << "No DT geometry is available."; 
  }

  // Build CSC layers
  edm::ESHandle<CSCGeometry> csc;
  record.getRecord<MuonGeometryRecord>().get(csc);
  if (csc.isValid()) {
    muonDetLayerGeometry->addCSCLayers(MuonCSCDetLayerGeometryBuilder::buildLayers(*csc));
  } else {
    LogInfo(metname) << "No CSC geometry is available.";
  }

  // Build GEM layers
  edm::ESHandle<GEMGeometry> gem;
  record.getRecord<MuonGeometryRecord>().get(gem);
  if (gem.isValid()) {
      muonDetLayerGeometry->addGEMLayers(MuonGEMDetLayerGeometryBuilder::buildEndcapLayers(*gem));
  } else {
     LogInfo(metname) << "No GEM geometry is available.";
  }


  // Build RPC layers
  edm::ESHandle<RPCGeometry> rpc;
  record.getRecord<MuonGeometryRecord>().get(rpc);
  if (rpc.isValid()) {
    muonDetLayerGeometry->addRPCLayers(MuonRPCDetLayerGeometryBuilder::buildBarrelLayers(*rpc),MuonRPCDetLayerGeometryBuilder::buildEndcapLayers(*rpc));
  } else {
    LogInfo(metname) << "No RPC geometry is available.";
  }  
  

  // Sort layers properly
  muonDetLayerGeometry->sortLayers();

  return boost::shared_ptr<MuonDetLayerGeometry>(muonDetLayerGeometry);
}
