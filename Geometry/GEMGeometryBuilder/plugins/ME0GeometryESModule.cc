/** \file
 *
 *  $Date: 2012/11/15 23:06:37 $
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometryBuilder/plugins/ME0GeometryESModule.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromCondDB.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>

#include <memory>

using namespace edm;

ME0GeometryESModule::ME0GeometryESModule(const edm::ParameterSet & p){
  comp11 = p.getUntrackedParameter<bool>("compatibiltyWith11",true);
  // Find out if using the DDD or CondDB Geometry source.
  useDDD = p.getUntrackedParameter<bool>("useDDD",true);
  setWhatProduced(this);

}


ME0GeometryESModule::~ME0GeometryESModule(){}


boost::shared_ptr<ME0Geometry>
ME0GeometryESModule::produce(const MuonGeometryRecord & record) {
  if(useDDD){
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    ME0GeometryBuilderFromDDD builder(comp11);
    return boost::shared_ptr<ME0Geometry>(builder.build(&(*cpv), *mdc));
  }else{
    edm::ESHandle<RecoIdealGeometry> rigme0;
    record.getRecord<ME0RecoGeometryRcd>().get(rigme0);
    ME0GeometryBuilderFromCondDB builder(comp11);
    return boost::shared_ptr<ME0Geometry>(builder.build(*rigme0));
  }

}

DEFINE_FWK_EVENTSETUP_MODULE(ME0GeometryESModule);
