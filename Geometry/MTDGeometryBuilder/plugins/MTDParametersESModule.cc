#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"

#include <memory>

class MTDParametersESModule : public edm::ESProducer {
public:
  MTDParametersESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<PMTDParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const PMTDParametersRcd&);

private:
  MTDParametersFromDD builder;

  const bool fromDD4hep_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddCompactToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
};

MTDParametersESModule::MTDParametersESModule(const edm::ParameterSet& pset)
    : fromDD4hep_(pset.getParameter<bool>("fromDD4hep")) {
  auto cc = setWhatProduced(this);
  if (!fromDD4hep_) {
    ddCompactToken_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  } else {
    dd4hepToken_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  }
}

void MTDParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("mtdParameters", desc);
}

MTDParametersESModule::ReturnType MTDParametersESModule::produce(const PMTDParametersRcd& iRecord) {
  auto ptp = std::make_unique<PMTDParameters>();
  if (!fromDD4hep_) {
    auto cpv = iRecord.getTransientHandle(ddCompactToken_);
    builder.build(cpv.product(), *ptp);
  } else {
    auto cpv = iRecord.getTransientHandle(dd4hepToken_);
    builder.build(cpv.product(), *ptp);
  }

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDParametersESModule);
