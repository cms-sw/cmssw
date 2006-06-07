/** \file GlobalTrackingGeometryESProducer.cc
 *
 *  $Date: 2006/05/10 18:05:36 $
 *  $Revision: 1.4 $
 *  \author Matteo Sani
 */

#include <Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryESProducer.h>
#include <Geometry/GlobalTrackingGeometryBuilder/src/GlobalTrackingGeometryBuilder.h>

#include <Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Framework/interface/NoProxyException.h>

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
      
  try {
  
    record.getRecord<TrackerDigiGeometryRecord>().get(tk);
    if (tk.isValid())
        LogDebug("GeometryGlobalTrackingGeometryBuilder") << "No valid Tracker geometry is available.";

  } catch (edm::eventsetup::NoProxyException<TrackerGeometry>& e) {
    // No Tk geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No Tracker geometry is available." << e.what();
  }    

  try {
  
    record.getRecord<MuonGeometryRecord>().get(dt);
    if (dt.isValid())
        LogDebug("GeometryGlobalTrackingGeometryBuilder") << "No valid DT geometry is available.";

  } catch (edm::eventsetup::NoProxyException<DTGeometry>& e) {
    // No DT geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No DT geometry is available." << e.what();
  }    

  try {
  
    record.getRecord<MuonGeometryRecord>().get(csc);
    if (csc.isValid())
        LogDebug("GeometryGlobalTrackingGeometryBuilder") << "No valid CSC geometry is available.";

  } catch (edm::eventsetup::NoProxyException<CSCGeometry>& e) {
    // No CSC geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No CSC geometry is available." << e.what();
  }    

  try {
  
    record.getRecord<MuonGeometryRecord>().get(rpc);
    if (rpc.isValid())
        LogDebug("GeometryGlobalTrackingGeometryBuilder") << "No valid RPC geometry is available.";

  } catch (edm::eventsetup::NoProxyException<RPCGeometry>& e) {
    // No RPC geo available
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No RPC geometry is available." << e.what();
  }    

  GlobalTrackingGeometryBuilder builder;
  return boost::shared_ptr<GlobalTrackingGeometry>(builder.build(&(*tk), &(*dt), &(*csc), &(*rpc)));
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalTrackingGeometryESProducer)
