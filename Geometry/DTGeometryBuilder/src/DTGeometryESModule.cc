/** \file
 *
 *  $Date: 2006/02/22 10:59:28 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/DTGeometryBuilder/src/DTGeometryESModule.h>
#include <Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h>

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>

#include <memory>

using namespace edm;

DTGeometryESModule::DTGeometryESModule(const edm::ParameterSet & p){
  setWhatProduced(this);
}


DTGeometryESModule::~DTGeometryESModule(){}


boost::shared_ptr<DTGeometry>
DTGeometryESModule::produce(const MuonGeometryRecord & record) {
  edm::ESHandle<DDCompactView> cpv;
  record.getRecord<IdealGeometryRecord>().get(cpv);
  DTGeometryBuilderFromDDD builder;
  return boost::shared_ptr<DTGeometry>(builder.build(&(*cpv)));
}

DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESModule)
