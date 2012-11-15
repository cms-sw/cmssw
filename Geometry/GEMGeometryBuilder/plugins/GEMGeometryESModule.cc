/** \file
 *
 *  $Date: 2010/03/25 22:08:44 $
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometryBuilder/plugins/GEMGeometryESModule.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <memory>

using namespace edm;

GEMGeometryESModule::GEMGeometryESModule(const edm::ParameterSet & p){
  comp11 = p.getUntrackedParameter<bool>("compatibiltyWith11",true);
  // Find out if using the DDD or CondDB Geometry source.
  useDDD = p.getUntrackedParameter<bool>("useDDD",true);
  setWhatProduced(this);

}


GEMGeometryESModule::~GEMGeometryESModule(){}


boost::shared_ptr<GEMGeometry>
GEMGeometryESModule::produce(const MuonGeometryRecord & record) {
  if(useDDD){
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    GEMGeometryBuilderFromDDD builder(comp11);
    return boost::shared_ptr<GEMGeometry>(builder.build(&(*cpv), *mdc));
  }else{
    edm::ESHandle<RecoIdealGeometry> riggem;
    record.getRecord<GEMRecoGeometryRcd>().get(riggem);
    GEMGeometryBuilderFromCondDB builder(comp11);
    return boost::shared_ptr<GEMGeometry>(builder.build(*riggem));
  }

}

DEFINE_FWK_EVENTSETUP_MODULE(GEMGeometryESModule);
