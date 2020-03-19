/*
//\class RPCGeometryESModule

 Description: RPC GeometryESModule from DD & DD4hep
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
*/

#include "Geometry/RPCGeometryBuilder/plugins/RPCGeometryESModule.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromCondDB.h"

#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <unordered_map>

#include <memory>

using namespace edm;

RPCGeometryESModule::RPCGeometryESModule(const edm::ParameterSet& p)
    : comp11_{p.getUntrackedParameter<bool>("compatibiltyWith11", true)},
      useDDD_{p.getUntrackedParameter<bool>("useDDD", true)},
      useDD4hep_{p.getUntrackedParameter<bool>("useDD4hep", false)} {
  auto cc = setWhatProduced(this);

  const edm::ESInputTag kEmptyTag;
  if (useDDD_) {
    idealGeomToken_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(kEmptyTag);
    dddConstantsToken_ = cc.consumesFrom<MuonDDDConstants, MuonNumberingRecord>(kEmptyTag);
  } else if (useDD4hep_) {
    idealDD4hepGeomToken_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(kEmptyTag);
    dd4hepConstantsToken_ = cc.consumesFrom<cms::MuonNumbering, MuonNumberingRecord>(kEmptyTag);
  } else {
    recoIdealToken_ = cc.consumesFrom<RecoIdealGeometry, RPCRecoGeometryRcd>(kEmptyTag);
  }
}

std::unique_ptr<RPCGeometry> RPCGeometryESModule::produce(const MuonGeometryRecord& record) {
  if (useDDD_) {
    edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle(idealGeomToken_);

    auto const& mdc = record.get(dddConstantsToken_);
    RPCGeometryBuilderFromDDD builder(comp11_);
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  } else if (useDD4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> cpv = record.getTransientHandle(idealDD4hepGeomToken_);

    auto const& mdc = record.get(dd4hepConstantsToken_);
    RPCGeometryBuilderFromDDD builder(comp11_);
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  } else {
    auto const& rigrpc = record.get(recoIdealToken_);
    RPCGeometryBuilderFromCondDB builder(comp11_);
    return std::unique_ptr<RPCGeometry>(builder.build(rigrpc));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
