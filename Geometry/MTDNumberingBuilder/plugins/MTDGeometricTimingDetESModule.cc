#include "Geometry/MTDNumberingBuilder/plugins/DDCmsMTDConstruction.h"
#include "Geometry/MTDNumberingBuilder/plugins/CondDBCmsMTDConstruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>

class MTDGeometricTimingDetESModule : public edm::ESProducer {
public:
  MTDGeometricTimingDetESModule(const edm::ParameterSet& p);
  ~MTDGeometricTimingDetESModule() override;

  std::unique_ptr<GeometricTimingDet> produce(const IdealGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool fromDDD_;
  const bool fromDD4hep_;

  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddCompactToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  edm::ESGetToken<PGeometricTimingDet, IdealGeometryRecord> pGTDetToken_;
};

using namespace edm;

MTDGeometricTimingDetESModule::MTDGeometricTimingDetESModule(const edm::ParameterSet& p)
    : fromDDD_(p.getParameter<bool>("fromDDD")), fromDD4hep_(p.getParameter<bool>("fromDD4hep")) {
  auto cc = setWhatProduced(this);
  if (fromDDD_) {
    ddCompactToken_ = cc.consumes<DDCompactView>(edm::ESInputTag());
  } else if (fromDD4hep_) {
    dd4hepToken_ = cc.consumes<cms::DDCompactView>(edm::ESInputTag());
  } else {
    pGTDetToken_ = cc.consumes<PGeometricTimingDet>(edm::ESInputTag());
  }
}

MTDGeometricTimingDetESModule::~MTDGeometricTimingDetESModule() {}

void MTDGeometricTimingDetESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription descDB;
  descDB.add<bool>("fromDDD", false);
  descDB.add<bool>("fromDD4hep", false);
  descriptions.add("mtdNumberingGeometryDB", descDB);

  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("mtdNumberingGeometry", desc);
}

std::unique_ptr<GeometricTimingDet> MTDGeometricTimingDetESModule::produce(const IdealGeometryRecord& iRecord) {
  if (fromDDD_) {
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(ddCompactToken_);
    return DDCmsMTDConstruction::construct((*cpv));
  } else if (fromDD4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(dd4hepToken_);
    return DDCmsMTDConstruction::construct((*cpv));
  } else {
    PGeometricTimingDet const& pgd = iRecord.get(pGTDetToken_);
    return CondDBCmsMTDConstruction::construct(pgd);
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDGeometricTimingDetESModule);
