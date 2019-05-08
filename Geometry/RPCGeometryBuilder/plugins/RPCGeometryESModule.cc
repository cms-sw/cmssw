/** \file
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/RPCGeometryBuilder/plugins/RPCGeometryESModule.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromCondDB.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <memory>

using namespace edm;

RPCGeometryESModule::RPCGeometryESModule(const edm::ParameterSet & p):
  comp11{p.getUntrackedParameter<bool>("compatibiltyWith11",true)},
  // Find out if using the DDD or CondDB Geometry source.
  useDDD{p.getUntrackedParameter<bool>("useDDD",true)}
{
  auto cc = setWhatProduced(this);

    const edm::ESInputTag kEmptyTag;
  if(useDDD) {
    idealGeomToken_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(kEmptyTag);
    dddConstantsToken_ = cc.consumesFrom<MuonDDDConstants, MuonNumberingRecord>(kEmptyTag);
  } else {
    recoIdealToken_ = cc.consumesFrom<RecoIdealGeometry, RPCRecoGeometryRcd>(kEmptyTag);
  }
}


std::unique_ptr<RPCGeometry>
RPCGeometryESModule::produce(const MuonGeometryRecord & record) {
  if(useDDD){
    edm::ESTransientHandle<DDCompactView> cpv = record.getTransientHandle( idealGeomToken_ ); 

    auto const& mdc = record.get( dddConstantsToken_ );
    RPCGeometryBuilderFromDDD builder(comp11);
    return std::unique_ptr<RPCGeometry>(builder.build(&(*cpv), mdc));
  }else{
    auto const& rigrpc = record.get( recoIdealToken_ );
    RPCGeometryBuilderFromCondDB builder(comp11);
    return std::unique_ptr<RPCGeometry>(builder.build(rigrpc));
  }

}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
