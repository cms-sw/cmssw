/** \file
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometryBuilder/plugins/GEMGeometryESModule.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromCondDB.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>

using namespace edm;

GEMGeometryESModule::GEMGeometryESModule(const edm::ParameterSet & p)
{
  useDDD = p.getParameter<bool>("useDDD");
  setWhatProduced(this);
}

GEMGeometryESModule::~GEMGeometryESModule(){}

std::unique_ptr<GEMGeometry>
GEMGeometryESModule::produce(const MuonGeometryRecord & record) 
{
  auto gemGeometry = std::make_unique<GEMGeometry>();

  if( useDDD ) {
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    GEMGeometryBuilderFromDDD builder;
    builder.build(*gemGeometry, &(*cpv), *mdc);
  } else {
    edm::ESHandle<RecoIdealGeometry> riggem;
    record.getRecord<GEMRecoGeometryRcd>().get(riggem);
    GEMGeometryBuilderFromCondDB builder;
    builder.build(*gemGeometry, *riggem);
  }
  
  return gemGeometry;  
}

DEFINE_FWK_EVENTSETUP_MODULE(GEMGeometryESModule);
