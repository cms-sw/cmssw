/** \file
 *
 *  $Date: 2010/03/25 22:08:44 $
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


boost::shared_ptr<RPCGeometry>
RPCGeometryESModule::produce(const MuonGeometryRecord & record) {

  std::cout<<"RPCGeometryESModule :: produce"<<std::endl;

  if(useDDD){
    edm::ESTransientHandle<DDCompactView> cpv;
    std::cout<<"RPCGeometryESModule :: produce :: 1/4 Get Muon Ideal Geometry Record"<<std::endl;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    std::cout<<"RPCGeometryESModule :: produce :: 2/4 Get Muon DDD Constants"<<std::endl;
    edm::ESHandle<MuonDDDConstants> mdc;
    std::cout<<"RPCGeometryESModule :: produce :: 3/4 Get Muon Numbering Record"<<std::endl;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    std::cout<<"RPCGeometryESModule :: produce :: 4/4 Build"<<std::endl;
    RPCGeometryBuilderFromDDD builder(comp11);
    return boost::shared_ptr<RPCGeometry>(builder.build(&(*cpv), *mdc));
  }else{
    edm::ESHandle<RecoIdealGeometry> rigrpc;
    record.getRecord<RPCRecoGeometryRcd>().get(rigrpc);
    RPCGeometryBuilderFromCondDB builder(comp11);
    return boost::shared_ptr<RPCGeometry>(builder.build(*rigrpc));
  }

}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
