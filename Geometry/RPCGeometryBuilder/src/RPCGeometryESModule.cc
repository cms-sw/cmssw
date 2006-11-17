/** \file
 *
 *  $Date: 2006/03/20 23:14:11 $
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/RPCGeometryBuilder/src/RPCGeometryESModule.h"
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <memory>

using namespace edm;

RPCGeometryESModule::RPCGeometryESModule(const edm::ParameterSet & p){
  setWhatProduced(this);
  comp11 = p.getUntrackedParameter<bool>("compatibiltyWith11",true);
}


RPCGeometryESModule::~RPCGeometryESModule(){}


boost::shared_ptr<RPCGeometry>
RPCGeometryESModule::produce(const MuonGeometryRecord & record) {
  edm::ESHandle<DDCompactView> cpv;
  record.getRecord<IdealGeometryRecord>().get(cpv);
	  RPCGeometryBuilderFromDDD builder(comp11);
  return boost::shared_ptr<RPCGeometry>(builder.build(&(*cpv)));
}

DEFINE_FWK_EVENTSETUP_MODULE(RPCGeometryESModule);
