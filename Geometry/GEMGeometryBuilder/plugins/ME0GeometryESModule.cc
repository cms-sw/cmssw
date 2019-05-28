/** \file
 *
 *  \author M. Maggi - INFN Bari
 */

#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromCondDB.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

using namespace edm;

class ME0GeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  ME0GeometryESModule(const edm::ParameterSet& p);

  /// Destructor
  ~ME0GeometryESModule() override;

  /// Produce ME0Geometry.
  std::unique_ptr<ME0Geometry> produce(const MuonGeometryRecord& record);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  edm::ESGetToken<MuonDDDConstants, MuonNumberingRecord> mdcToken_;
  edm::ESGetToken<RecoIdealGeometry, ME0RecoGeometryRcd> rigme0Token_;
  // use the DDD as Geometry source
  bool useDDD_;
};

ME0GeometryESModule::ME0GeometryESModule(const edm::ParameterSet& p) {
  useDDD_ = p.getParameter<bool>("useDDD");
  auto cc = setWhatProduced(this);
  if (useDDD_) {
    cc.setConsumes(cpvToken_).setConsumes(mdcToken_);
  } else {
    cc.setConsumes(rigme0Token_);
  }
}

ME0GeometryESModule::~ME0GeometryESModule() {}

std::unique_ptr<ME0Geometry> ME0GeometryESModule::produce(const MuonGeometryRecord& record) {
  LogTrace("ME0GeometryESModule") << "ME0GeometryESModule::produce with useDDD = " << useDDD_;

  if (useDDD_) {
    LogTrace("ME0GeometryESModule") << "ME0GeometryESModule::produce :: ME0GeometryBuilderFromDDD builder";
    auto cpv = record.getTransientHandle(cpvToken_);
    const auto& mdc = record.get(mdcToken_);
    ME0GeometryBuilderFromDDD builder;
    return std::unique_ptr<ME0Geometry>(builder.build(cpv.product(), mdc));
  } else {
    LogTrace("ME0GeometryESModule") << "ME0GeometryESModule::produce :: ME0GeometryBuilderFromCondDB builder";
    const auto& rigme0 = record.get(rigme0Token_);
    ME0GeometryBuilderFromCondDB builder;
    return std::unique_ptr<ME0Geometry>(builder.build(rigme0));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(ME0GeometryESModule);
