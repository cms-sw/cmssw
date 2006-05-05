#include "Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryESProducer.h"
#include "Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryBuilder.h"

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>

using namespace edm;

GlobalTrackingGeometryESProducer::GlobalTrackingGeometryESProducer(const edm::ParameterSet & p){

  setWhatProduced(this);
  
  // FIXME: set the parameters
}

GlobalTrackingGeometryESProducer::~GlobalTrackingGeometryESProducer(){}

boost::shared_ptr<GlobalTrackingGeometry>
GlobalTrackingGeometryESProducer::produce(const GlobalTrackingGeometryRecord& record) {

  // DO NOT CHANGE THE ORDER OF THE GEOMETRIES !!!!!!!    
  edm::ESHandle<TrackerGeometry> tk;
  edm::ESHandle<DTGeometry> dt;
  edm::ESHandle<CSCGeometry> csc;
  edm::ESHandle<RPCGeometry> rpc;

  record.getRecord<GlobalTrackingGeometryRecord>().get(tk);
  record.getRecord<GlobalTrackingGeometryRecord>().get(dt);
  record.getRecord<GlobalTrackingGeometryRecord>().get(csc);
  record.getRecord<GlobalTrackingGeometryRecord>().get(rpc);

  GlobalTrackingGeometryBuilder builder;
  return boost::shared_ptr<GlobalTrackingGeometry>(builder.build(&(*tk),&(*dt),&(*csc),&(*rpc)));
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalTrackingGeometryESProducer)
