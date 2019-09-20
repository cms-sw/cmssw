#ifndef RPCGeometry_RPCGeometryESModule_h
#define RPCGeometry_RPCGeometryESModule_h

/*
//\class RPCGeometryESModule

 Description: RPC GeometryESModule from DD & DD4hep
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  Fri, 20 Sep 2019 
*/

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/DDCMS/interface/DDCompactView.h>

#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include <memory>

class RPCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  RPCGeometryESModule(const edm::ParameterSet& p);

  /// Destructor
  ~RPCGeometryESModule() override = default;

  /// Produce RPCGeometry.
  std::unique_ptr<RPCGeometry> produce(const MuonGeometryRecord& record);

private:
  //Used without DDD
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> idealGeomToken_;
  edm::ESGetToken<MuonDDDConstants, MuonNumberingRecord> dddConstantsToken_;
  // dd4hep
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> idealDD4hepGeomToken_;
  edm::ESGetToken<cms::MuonNumbering, MuonNumberingRecord> dd4hepConstantsToken_;

  //Used with DDD
  edm::ESGetToken<RecoIdealGeometry, RPCRecoGeometryRcd> recoIdealToken_;

  const bool comp11_;
  const bool useDDD_;
  const bool useDD4hep_;
};
#endif
