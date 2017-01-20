/** \file
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometryBuilder/plugins/ME0GeometryESModule.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromDDD10EtaPart.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromCondDB.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>

using namespace edm;

ME0GeometryESModule::ME0GeometryESModule(const edm::ParameterSet & p)
{
  useDDD       = p.getParameter<bool>("useDDD");
  use10EtaPart = p.getParameter<bool>("use10EtaPart");
  setWhatProduced(this);
}


ME0GeometryESModule::~ME0GeometryESModule(){}


std::shared_ptr<ME0Geometry>
ME0GeometryESModule::produce(const MuonGeometryRecord & record) 
{

  LogTrace("ME0GeometryESModule")<<"ME0GeometryESModule::produce with useDDD = "<<useDDD<<" and use10EtaPart = "<<use10EtaPart;

  if(useDDD && !use10EtaPart){
    LogTrace("ME0GeometryESModule")<<"ME0GeometryESModule::produce :: ME0GeometryBuilderFromDDD builder";
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    ME0GeometryBuilderFromDDD builder;
    return std::shared_ptr<ME0Geometry>(builder.build(&(*cpv), *mdc));
  }
  else if(useDDD && use10EtaPart){
    LogTrace("ME0GeometryESModule")<<"ME0GeometryESModule::produce :: ME0GeometryBuilderFromDDD10EtaPart builder";
    edm::ESTransientHandle<DDCompactView> cpv;
    record.getRecord<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    record.getRecord<MuonNumberingRecord>().get(mdc);
    ME0GeometryBuilderFromDDD10EtaPart builder;
    return std::shared_ptr<ME0Geometry>(builder.build(&(*cpv), *mdc));
  }
  else{
    LogTrace("ME0GeometryESModule")<<"ME0GeometryESModule::produce :: ME0GeometryBuilderFromCondDB builder";
    edm::ESHandle<RecoIdealGeometry> rigme0;
    record.getRecord<ME0RecoGeometryRcd>().get(rigme0);
    ME0GeometryBuilderFromCondDB builder;
    return std::shared_ptr<ME0Geometry>(builder.build(*rigme0));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(ME0GeometryESModule);
