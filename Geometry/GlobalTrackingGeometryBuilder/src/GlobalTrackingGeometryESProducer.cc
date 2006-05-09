/** \file GlobalTrackingGeometryESProducer.cc
 *
 *  $Date: 2006/05/06 13:46:16 $
 *  $Revision: 1.1 $
 *  \author Matteo Sani
 */

#include "Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryESProducer.h"
#include "Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryBuilder.h"

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
      
  try {
  
    record.getRecord<GlobalTrackingGeometryRecord>().get(tk);
  } catch (...) {
    // No Tk geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No Tracker geometry is available.";
  }    

  try {
  
    record.getRecord<GlobalTrackingGeometryRecord>().get(dt);
  } catch (...) {
    // No DT geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No DT geometry is available.";
  }    

  try {
  
    record.getRecord<GlobalTrackingGeometryRecord>().get(csc);
  } catch (...) {
    // No CSC geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No CSC geometry is available.";
  }    

  try {
  
    record.getRecord<GlobalTrackingGeometryRecord>().get(rpc);
  } catch (...) {
    // No RPC geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No RPC geometry is available.";
  }    

  GlobalTrackingGeometryBuilder builder;
  return boost::shared_ptr<GlobalTrackingGeometry>(builder.build(&(*tk),&(*dt),&(*csc),&(*rpc)));
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalTrackingGeometryESProducer)
