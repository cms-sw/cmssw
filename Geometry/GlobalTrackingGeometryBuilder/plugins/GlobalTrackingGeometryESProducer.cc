/** \file GlobalTrackingGeometryESProducer.cc
 *
 *  \author Matteo Sani
 */

#include "Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryESProducer.h"
#include "Geometry/GlobalTrackingGeometryBuilder/plugins/GlobalTrackingGeometryBuilder.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/NoProxyException.h"
#include "FWCore/Framework/interface/NoRecordException.h"

#include <memory>

using namespace edm;

GlobalTrackingGeometryESProducer::GlobalTrackingGeometryESProducer(const edm::ParameterSet & p){
  setWhatProduced(this);
}

GlobalTrackingGeometryESProducer::~GlobalTrackingGeometryESProducer(){}

std::shared_ptr<GlobalTrackingGeometry>
GlobalTrackingGeometryESProducer::produce(const GlobalTrackingGeometryRecord& record) {

  TrackerGeometry const* tk = nullptr;
  MTDGeometry const* mtd = nullptr;
  DTGeometry const* dt = nullptr;
  CSCGeometry const* csc = nullptr;
  RPCGeometry const* rpc = nullptr;
  GEMGeometry const* gem = nullptr;
  ME0Geometry const* me0 = nullptr;

  try {
    edm::ESHandle<TrackerGeometry> tkH;
    record.getRecord<TrackerDigiGeometryRecord>().get(tkH);
    if(tkH.isValid()) {
      tk = tkH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No Tracker geometry is available.";
    }
  } catch (edm::eventsetup::NoRecordException<TrackerDigiGeometryRecord>& e){
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No TrackerDigiGeometryRecord is available.";    
  }

  try {
    edm::ESHandle<MTDGeometry> mtdH;
    record.getRecord<MTDDigiGeometryRecord>().get(mtdH);
    if(mtdH.isValid()) {
      mtd = mtdH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No MTD geometry is available.";
    }
  } catch (edm::eventsetup::NoRecordException<MTDDigiGeometryRecord>& e){
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No MTDDigiGeometryRecord is available.";    
  }


  try {
    edm::ESHandle<DTGeometry> dtH;
    record.getRecord<MuonGeometryRecord>().get(dtH);
    if(dtH.isValid()) {
      dt = dtH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No DT geometry is available.";
    }

    edm::ESHandle<CSCGeometry> cscH;
    record.getRecord<MuonGeometryRecord>().get(cscH);
    if(cscH.isValid()) {
      csc = cscH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No CSC geometry is available.";
    }
    
    edm::ESHandle<RPCGeometry> rpcH;
    record.getRecord<MuonGeometryRecord>().get(rpcH);
    if(rpcH.isValid()) {
      rpc = rpcH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No RPC geometry is available.";
    }

    edm::ESHandle<GEMGeometry> gemH;
    record.getRecord<MuonGeometryRecord>().get(gemH);
    if(gemH.isValid()) {
      gem = gemH.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No GEM geometry is available.";
    }

    edm::ESHandle<ME0Geometry> me0H;
    record.getRecord<MuonGeometryRecord>().get(me0H);
    if(me0H.isValid()) {
      me0 = me0H.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No ME0 geometry is available.";
    }

  } catch (edm::eventsetup::NoRecordException<MuonGeometryRecord>& e){
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No MuonGeometryRecord is available.";    
  }

  GlobalTrackingGeometryBuilder builder;
  return std::shared_ptr<GlobalTrackingGeometry>(builder.build(tk, mtd, dt, csc, rpc, gem, me0));
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalTrackingGeometryESProducer);
