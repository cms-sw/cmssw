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

#include <memory>

using namespace edm;

GlobalTrackingGeometryESProducer::GlobalTrackingGeometryESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);
  trackerToken_ = cc.consumesFrom<TrackerGeometry, TrackerDigiGeometryRecord>(edm::ESInputTag{});
  mtdToken_ = cc.consumesFrom<MTDGeometry, MTDDigiGeometryRecord>(edm::ESInputTag{});
  dtToken_ = cc.consumesFrom<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{});
  cscToken_ = cc.consumesFrom<CSCGeometry, MuonGeometryRecord>(edm::ESInputTag{});
  rpcToken_ = cc.consumesFrom<RPCGeometry, MuonGeometryRecord>(edm::ESInputTag{});
  gemToken_ = cc.consumesFrom<GEMGeometry, MuonGeometryRecord>(edm::ESInputTag{});
  me0Token_ = cc.consumesFrom<ME0Geometry, MuonGeometryRecord>(edm::ESInputTag{});
}

GlobalTrackingGeometryESProducer::~GlobalTrackingGeometryESProducer() {}

std::unique_ptr<GlobalTrackingGeometry> GlobalTrackingGeometryESProducer::produce(
    const GlobalTrackingGeometryRecord& record) {
  TrackerGeometry const* tk = nullptr;
  MTDGeometry const* mtd = nullptr;
  DTGeometry const* dt = nullptr;
  CSCGeometry const* csc = nullptr;
  RPCGeometry const* rpc = nullptr;
  GEMGeometry const* gem = nullptr;
  ME0Geometry const* me0 = nullptr;

  if (auto tkRecord = record.tryToGetRecord<TrackerDigiGeometryRecord>()) {
    if (auto tkH = tkRecord->getHandle(trackerToken_)) {
      tk = tkH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No Tracker geometry is available.";
    }
  } else {
    LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No TrackerDigiGeometryRecord is available.";
  }

  if (auto mtdRecord = record.tryToGetRecord<MTDDigiGeometryRecord>()) {
    if (auto mtdH = mtdRecord->getHandle(mtdToken_)) {
      mtd = mtdH.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No MTD geometry is available.";
    }
  } else {
    LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No MTDDigiGeometryRecord is available.";
  }

  if (auto muonRecord = record.tryToGetRecord<MuonGeometryRecord>()) {
    if (auto dtH = muonRecord->getHandle(dtToken_)) {
      dt = dtH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No DT geometry is available.";
    }

    if (auto cscH = muonRecord->getHandle(cscToken_)) {
      csc = cscH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No CSC geometry is available.";
    }

    if (auto rpcH = muonRecord->getHandle(rpcToken_)) {
      rpc = rpcH.product();
    } else {
      LogWarning("GeometryGlobalTrackingGeometryBuilder") << "No RPC geometry is available.";
    }

    if (auto gemH = muonRecord->getHandle(gemToken_)) {
      gem = gemH.product();
    } else {
      LogInfo("GeometryGlobalTrackingGeometryBuilder") << "No GEM geometry is available.";
    }

    if (auto me0H = muonRecord->getHandle(me0Token_)) {
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
