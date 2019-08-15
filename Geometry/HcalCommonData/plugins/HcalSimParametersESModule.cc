#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "Geometry/HcalCommonData/interface/HcalSimParametersFromDD.h"

#include <memory>

class HcalSimParametersESModule : public edm::ESProducer {
public:
  HcalSimParametersESModule(const edm::ParameterSet&);
  ~HcalSimParametersESModule(void) override;

  using ReturnType = std::unique_ptr<HcalSimulationParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalParametersRcd&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
};

HcalSimParametersESModule::HcalSimParametersESModule(const edm::ParameterSet&)
    : cpvToken_{setWhatProduced(this).consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{})} {
  edm::LogInfo("HCAL") << "HcalSimParametersESModule::HcalSimParametersESModule";
}

HcalSimParametersESModule::~HcalSimParametersESModule() {}

void HcalSimParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hcalSimulationParameters", desc);
}

HcalSimParametersESModule::ReturnType HcalSimParametersESModule::produce(const HcalParametersRcd& iRecord) {
  edm::LogInfo("HcalESModule") << "HcalSimParametersESModule::produce(const HcalParametersRcd& iRecord)";
  edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvToken_);

  auto ptp = std::make_unique<HcalSimulationParameters>();
  HcalSimParametersFromDD builder;
  builder.build(&(*cpv), *ptp);

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalSimParametersESModule);
