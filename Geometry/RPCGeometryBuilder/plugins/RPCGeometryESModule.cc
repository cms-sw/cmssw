/*
//\class RPCGeometryESModule

Description: RPC GeometryESModule from DD & DD4hep
DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
//          Modified: Fri, 29 May 2020, following what Sunanda Banerjee made in PR #29842 PR #29943 and Ianna Osborne in PR #29954      
*/
#include "Geometry/RPCGeometryBuilder/plugins/RPCGeometryESModule.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilder.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromCondDB.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
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

  if (useDDD_) {
    cc.setConsumes(idealGeomToken_).setConsumes(dddConstantsToken_);
  } else if (useDD4hep_) {
    cc.setConsumes(idealDD4hepGeomToken_).setConsumes(dddConstantsToken_);
  } else {
    cc.setConsumes(recoIdealToken_);
  }
}

std::unique_ptr<RPCGeometry> RPCGeometryESModule::produce(const MuonGeometryRecord& record) {
  if (useDDD_) {
    edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle(idealGeomToken_);
    auto const& mdc = record.get(dddConstantsToken_);
    RPCGeometryBuilder builder(comp11_);
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  } else if (useDD4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> cpv = record.getTransientHandle(idealDD4hepGeomToken_);
    auto const& mdc = record.get(dddConstantsToken_);
    RPCGeometryBuilder builder(comp11_);
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  } else {
    auto const& rigrpc = record.get(recoIdealToken_);
    RPCGeometryBuilderFromCondDB builder(comp11_);
    return std::unique_ptr<RPCGeometry>(builder.build(rigrpc));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
