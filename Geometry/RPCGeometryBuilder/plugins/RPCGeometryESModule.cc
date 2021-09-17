/*
//\class RPCGeometryESModule

Description: RPC GeometryESModule from DD & DD4hep
DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
//          Modified: Fri, 29 May 2020, following what Sunanda Banerjee made in PR #29842 PR #29943 and Ianna Osborne in PR #29954      
*/
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilder.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromCondDB.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <unordered_map>
#include <memory>

class RPCGeometryESModule : public edm::ESProducer {
public:
  RPCGeometryESModule(const edm::ParameterSet& p);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  std::unique_ptr<RPCGeometry> produce(const MuonGeometryRecord& record);

private:
  //DDD
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> idealGeomToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> dddConstantsToken_;
  // dd4hep
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> idealDD4hepGeomToken_;
  // Reco
  edm::ESGetToken<RecoIdealGeometry, RPCRecoGeometryRcd> recoIdealToken_;

  const bool fromDDD_;
  const bool fromDD4hep_;
};

RPCGeometryESModule::RPCGeometryESModule(const edm::ParameterSet& p)
    : fromDDD_{p.getUntrackedParameter<bool>("fromDDD", true)},
      fromDD4hep_{p.getUntrackedParameter<bool>("fromDD4hep", false)} {
  auto cc = setWhatProduced(this);

  if (fromDDD_) {
    idealGeomToken_ = cc.consumes();
    dddConstantsToken_ = cc.consumes();
  } else if (fromDD4hep_) {
    idealDD4hepGeomToken_ = cc.consumes();
    dddConstantsToken_ = cc.consumes();
  } else {
    recoIdealToken_ = cc.consumes();
  }
}

void RPCGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("fromDDD", true);
  desc.addUntracked<bool>("fromDD4hep", false);
  descriptions.add("RPCGeometryESModule", desc);
}

std::unique_ptr<RPCGeometry> RPCGeometryESModule::produce(const MuonGeometryRecord& record) {
  if (fromDDD_) {
    edm::LogVerbatim("RPCGeoemtryESModule") << "(0) RPCGeometryESModule  - DDD ";
    edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle(idealGeomToken_);
    auto const& mdc = record.get(dddConstantsToken_);
    RPCGeometryBuilder builder;
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  } else if (fromDD4hep_) {
    edm::LogVerbatim("RPCGeoemtryESModule") << "(0) RPCGeometryESModule  - DD4HEP ";
    edm::ESTransientHandle<cms::DDCompactView> cpv = record.getTransientHandle(idealDD4hepGeomToken_);
    auto const& mdc = record.get(dddConstantsToken_);
    RPCGeometryBuilder builder;
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  } else {
    edm::LogVerbatim("RPCGeoemtryESModule") << "(0) RPCGeometryESModule  - DB ";
    auto const& rigrpc = record.get(recoIdealToken_);
    RPCGeometryBuilderFromCondDB builder;
    return std::unique_ptr<RPCGeometry>(builder.build(rigrpc));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
