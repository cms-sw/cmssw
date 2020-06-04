#ifndef RPCGeometry_RPCGeometryESModule_h
#define RPCGeometry_RPCGeometryESModule_h
/*
//\class RPCGeometryESModule

Description: RPC GeometryESModule from DD & DD4hep
DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
//          Modified: Fri, 29 May 2020, following what Sunanda Banerjee made in PR #29842 PR #29943 and Ianna Osborne in PR #29954      
*/
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/MuonNumbering/interface/MuonGeometryConstants.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/DDCMS/interface/DDCompactView.h>
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include <memory>

class RPCGeometryESModule : public edm::ESProducer {
public:
  RPCGeometryESModule(const edm::ParameterSet& p);
  ~RPCGeometryESModule() override = default;
  std::unique_ptr<RPCGeometry> produce(const MuonGeometryRecord& record);

private:
  //DDD
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> idealGeomToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> dddConstantsToken_;
  // dd4hep
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> idealDD4hepGeomToken_;
  // Reco
  edm::ESGetToken<RecoIdealGeometry, RPCRecoGeometryRcd> recoIdealToken_;

  const bool useDDD_;
  const bool useDD4hep_;
};
#endif
