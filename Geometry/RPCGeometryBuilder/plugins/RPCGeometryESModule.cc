/** \file
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/RPCGeometryBuilder/plugins/RPCGeometryESModule.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromCondDB.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <memory>

using namespace edm;

RPCGeometryESModule::RPCGeometryESModule(const edm::ParameterSet & p){
  comp11 = p.getUntrackedParameter<bool>("compatibiltyWith11",true);
  // Find out if using the DDD or CondDB Geometry source.
  useDDD = p.getUntrackedParameter<bool>("useDDD",true);
  setWhatProduced(this);

}


RPCGeometryESModule::~RPCGeometryESModule(){}


std::unique_ptr<RPCGeometry>
RPCGeometryESModule::produce(const MuonGeometryRecord & record) {
  if(useDDD){
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    RPCGeometryBuilderFromDDD builder(comp11);
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), *mdc));
  }else{
    edm::ESHandle<RecoIdealGeometry> rigrpc;
    record.getRecord<RPCRecoGeometryRcd>().get(rigrpc);
    RPCGeometryBuilderFromCondDB builder(comp11);
    return std::unique_ptr<RPCGeometry>(builder.build(*rigrpc));
  }

}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
