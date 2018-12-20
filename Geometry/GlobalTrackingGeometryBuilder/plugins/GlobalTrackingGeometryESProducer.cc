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

std::unique_ptr<GlobalTrackingGeometry>
GlobalTrackingGeometryESProducer::produce(const GlobalTrackingGeometryRecord& record) {

  TrackerGeometry const* tk = nullptr;
  MTDGeometry const* mtd = nullptr;
  DTGeometry const* dt = nullptr;
  CSCGeometry const* csc = nullptr;
  RPCGeometry const* rpc = nullptr;
  GEMGeometry const* gem = nullptr;
  ME0Geometry const* me0 = nullptr;

  edm::ESHandle<TrackerGeometry> tkH;
  if( auto tkRecord = record.tryToGetRecord<TrackerDigiGeometryRecord>() ) {
    tkRecord->get(tkH);
    if(tkH.isValid()) {
      tk = tkH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No Tracker geometry is available.";
    }
  } else {
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No TrackerDigiGeometryRecord is available.";    
  }

  edm::ESHandle<MTDGeometry> mtdH;
  if( auto mtdRecord = record.tryToGetRecord<MTDDigiGeometryRecord>() ) {
    mtdRecord->get(mtdH);
    if( mtdH.isValid() ) {
      mtd = mtdH.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No MTD geometry is available.";
    }
  } else {
     LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No MTDDigiGeometryRecord is available.";
  }
  
  edm::ESHandle<DTGeometry> dtH;
  edm::ESHandle<CSCGeometry> cscH;
  edm::ESHandle<RPCGeometry> rpcH;
  edm::ESHandle<GEMGeometry> gemH;
  edm::ESHandle<ME0Geometry> me0H;
  if( auto muonRecord = record.tryToGetRecord<MuonGeometryRecord>() ) {
    muonRecord->get(dtH);
    if(dtH.isValid()) {
      dt = dtH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No DT geometry is available.";
    }
    
    muonRecord->get(cscH);
    if(cscH.isValid()) {
      csc = cscH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No CSC geometry is available.";
    }    
    
    muonRecord->get(rpcH);
    if(rpcH.isValid()) {
      rpc = rpcH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No RPC geometry is available.";
    }
    
    muonRecord->get(gemH);
    if(gemH.isValid()) {
      gem = gemH.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No GEM geometry is available.";
    }
    
    muonRecord->get(me0H);
    if(me0H.isValid()) {
      me0 = me0H.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No ME0 geometry is available.";
    }

  } else {
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No MuonGeometryRecord is available.";    
  }

  GlobalTrackingGeometryBuilder builder;
  return std::unique_ptr<GlobalTrackingGeometry>(builder.build(tk, mtd, dt, csc, rpc, gem, me0));
}

DEFINE_FWK_EVENTSETUP_MODULE(GlobalTrackingGeometryESProducer);
