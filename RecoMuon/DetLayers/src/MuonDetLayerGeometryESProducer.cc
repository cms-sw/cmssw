/** \file
 *
 *  $Date: 2006/04/25 17:03:23 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/src/MuonDetLayerGeometryESProducer.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>

#include <RecoMuon/DetLayers/src/MuonCSCDetLayerGeometryBuilder.h>

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

  //pair<vector<MuRingForwardLayer*>, vector<MuRingForwardLayer*> > csclayers; 
  pair<vector<DetLayer*>, vector<DetLayer*> > csclayers; 
  vector<MuRingForwardLayer*> dtlayers, rpclayers;
  
  try {
    edm::ESHandle<DTGeometry>  dt;
    record.getRecord<MuonGeometryRecord>().get(dt);
    if (dt.isValid()) {
        //dtlayers = MuonCSCDetLayerGeometryBuilder::buildLayers(*csc);
    }
  
  } catch (...) {
    // No DT geo available
    LogInfo("xxx") << "No DT geometry is available.";
  }  

  // Build CSC layers
  try {
    edm::ESHandle<CSCGeometry> csc;
    record.getRecord<MuonGeometryRecord>().get(csc);
    if (csc.isValid()) {
        csclayers = MuonCSCDetLayerGeometryBuilder::buildLayers(*csc);
    }
  } catch(...) {
    // No CSC geo available
    LogInfo("xxx") << "No CSC geometry is available.";
  }
  
  try {
    edm::ESHandle<RPCGeometry> rpc;
    record.getRecord<MuonGeometryRecord>().get(rpc);
    if (rpc.isValid()) {
        //rpclayers = MuonCSCDetLayerGeometryBuilder::buildLayers(*csc);
    }
  
  } catch (...) {
    // No RPC geo available
    LogInfo("xxx") << "No RPC geometry is available.";
  }  

  return boost::shared_ptr<MuonDetLayerGeometry>(new MuonDetLayerGeometry(csclayers));
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonDetLayerGeometryESProducer)
