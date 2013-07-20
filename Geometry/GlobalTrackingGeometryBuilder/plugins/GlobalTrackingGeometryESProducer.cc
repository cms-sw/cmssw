/** \file GlobalTrackingGeometryESProducer.cc
 *
 *  $Date: 2013/05/24 07:44:00 $
 *  $Revision: 1.2 $
 *  \author Matteo Sani
 */

#include <Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryESProducer.h>
#include <Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryBuilder.h>

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>

#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Framework/interface/NoProxyException.h>
#include <FWCore/Framework/interface/NoRecordException.h>

#include <memory>

using namespace edm;

GlobalTrackingGeometryESProducer::GlobalTrackingGeometryESProducer(const edm::ParameterSet & p){
  setWhatProduced(this);
}

GlobalTrackingGeometryESProducer::~GlobalTrackingGeometryESProducer(){}

boost::shared_ptr<GlobalTrackingGeometry>
GlobalTrackingGeometryESProducer::produce(const GlobalTrackingGeometryRecord& record) {

  edm::ESHandle<TrackerGeometry> tk;
  edm::ESHandle<DTGeometry> dt;
  edm::ESHandle<CSCGeometry> csc;
  edm::ESHandle<RPCGeometry> rpc;
  edm::ESHandle<GEMGeometry> gem;
      
  try {
    record.getRecord<TrackerDigiGeometryRecord>().get(tk);
  } catch (edm::eventsetup::NoProxyException<TrackerGeometry>& e) {
    // No Tk geo available
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No Tracker geometry is available.";
  } catch (edm::eventsetup::NoRecordException<TrackerDigiGeometryRecord>& e){
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No TrackerDigiGeometryRecord is available.";    
  }


  try {
    try {  
      record.getRecord<MuonGeometryRecord>().get(dt);
    } catch (edm::eventsetup::NoProxyException<DTGeometry>& e) {
      // No DT geo available
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No DT geometry is available.";
    } 

    try {
      record.getRecord<MuonGeometryRecord>().get(csc);
    } catch (edm::eventsetup::NoProxyException<CSCGeometry>& e) {
      // No CSC geo available
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No CSC geometry is available.";
    }    
    
    try {
      record.getRecord<MuonGeometryRecord>().get(rpc);      
    } catch (edm::eventsetup::NoProxyException<RPCGeometry>& e) {
      // No RPC geo available
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No RPC geometry is available.";
    }

    try {
      record.getRecord<MuonGeometryRecord>().get(gem);      
    } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
      // No GEM geo available
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No GEM geometry is available.";
    }

  } catch (edm::eventsetup::NoRecordException<MuonGeometryRecord>& e){
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No MuonGeometryRecord is available.";    
  }
  

  GlobalTrackingGeometryBuilder builder;
  return boost::shared_ptr<GlobalTrackingGeometry>(builder.build(&(*tk), &(*dt), &(*csc), &(*rpc), &(*gem)));
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalTrackingGeometryESProducer);
