/*
//\class ME0GeometryESModule

 Description: ME0 GeometryESModule from DD & DD4hep
              DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osborne made for DTs (DD4HEP migration)
//          Created:  29 Apr 2020 
*/

#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilder.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromCondDB.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <memory>

class ME0GeometryESModule : public edm::ESProducer {
public:
  ME0GeometryESModule(const edm::ParameterSet& p);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  std::unique_ptr<ME0Geometry> produce(const MuonGeometryRecord& record);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<MuonGeometryConstants, IdealGeometryRecord> mdcToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepcpvToken_;
  edm::ESGetToken<RecoIdealGeometry, ME0RecoGeometryRcd> rigme0Token_;
  // use the DDD or DD4hep as Geometry source
  bool fromDDD_;
  bool fromDD4hep_;
};

ME0GeometryESModule::ME0GeometryESModule(const edm::ParameterSet& p) {
  fromDDD_ = p.getParameter<bool>("fromDDD");
  fromDD4hep_ = p.getParameter<bool>("fromDD4hep");
  auto cc = setWhatProduced(this);
  if (fromDDD_) {
    cpvToken_ = cc.consumes();
    mdcToken_ = cc.consumes();
  } else if (fromDD4hep_) {
    dd4hepcpvToken_ = cc.consumes();
    mdcToken_ = cc.consumes();
  } else {
    rigme0Token_ = cc.consumes();
  }
  edm::LogVerbatim("GEMGeometry") << "ME0GeometryESModule::initailized with flags " << fromDDD_ << ":" << fromDD4hep_;
}

void ME0GeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("me0Geometry", desc);
}

std::unique_ptr<ME0Geometry> ME0GeometryESModule::produce(const MuonGeometryRecord& record) {
  edm::LogVerbatim("GEMGeometry") << "ME0GeometryESModule::produce with fromDDD = " << fromDDD_ << " fromDD4hep "
                                  << fromDD4hep_;
  if (fromDDD_) {
    edm::LogVerbatim("GEMGeometry") << "ME0GeometryESModule::produce :: ME0GeometryBuilder builder";
    auto cpv = record.getTransientHandle(cpvToken_);
    const auto& mdc = record.get(mdcToken_);
    ME0GeometryBuilder builder;
    return std::unique_ptr<ME0Geometry>(builder.build(cpv.product(), mdc));
  } else if (fromDD4hep_) {
    edm::LogVerbatim("GEMGeometry") << "ME0GeometryESModule::produce :: ME0GeometryBuilder builder DD4hep";
    auto cpv = record.getTransientHandle(dd4hepcpvToken_);
    const auto& mdc = record.get(mdcToken_);
    ME0GeometryBuilder builder;
    return std::unique_ptr<ME0Geometry>(builder.build(cpv.product(), mdc));
  } else {
    edm::LogVerbatim("GEMGeometry") << "ME0GeometryESModule::produce :: ME0GeometryBuilderFromCondDB builder";
    const auto& rigme0 = record.get(rigme0Token_);
    ME0GeometryBuilderFromCondDB builder;
    return std::unique_ptr<ME0Geometry>(builder.build(rigme0));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(ME0GeometryESModule);
