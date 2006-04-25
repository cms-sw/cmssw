/** \file
 *
 *  $Date: 2006/04/20 16:57:52 $
 *  $Revision: 1.1 $
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

#include <memory>

using namespace edm;

MuonDetLayerGeometryESProducer::MuonDetLayerGeometryESProducer(const edm::ParameterSet & p){
  setWhatProduced(this);
}


MuonDetLayerGeometryESProducer::~MuonDetLayerGeometryESProducer(){}


boost::shared_ptr<MuonDetLayerGeometry>
MuonDetLayerGeometryESProducer::produce(const MuonRecoGeometryRecord & record) {

  try {
    edm::ESHandle<DTGeometry>  dt;
    record.getRecord<MuonGeometryRecord>().get(dt);
  } catch (...) {
    // No DT geo available
  }  

  // Build CSC layers
  edm::ESHandle<CSCGeometry> csc;
  record.getRecord<MuonGeometryRecord>().get(csc);
  if (csc.isValid()) {
    vector<MuRingForwardLayer*> csclayers = MuonCSCDetLayerGeometryBuilder::buildLayers(*csc);
  }
  
  edm::ESHandle<RPCGeometry> rpc;
  record.getRecord<MuonGeometryRecord>().get(rpc);

  

  return boost::shared_ptr<MuonDetLayerGeometry>(new MuonDetLayerGeometry());
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonDetLayerGeometryESProducer)
