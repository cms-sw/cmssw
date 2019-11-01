
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "Geometry/HcalCommonData/interface/CaloSimParametersFromDD.h"

#include <memory>

//#define EDM_ML_DEBUG

class CaloSimParametersESModule : public edm::ESProducer {
public:
  CaloSimParametersESModule(const edm::ParameterSet&);
  ~CaloSimParametersESModule(void) override;

  using ReturnType = std::unique_ptr<CaloSimulationParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalParametersRcd&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  bool fromDD4Hep_;
};

CaloSimParametersESModule::CaloSimParametersESModule(const edm::ParameterSet& ps)
    : cpvToken_{setWhatProduced(this).consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{})} {
  fromDD4Hep_ = ps.getParameter<bool>("fromDD4Hep");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::CaloSimParametersESModule";
#endif
}

CaloSimParametersESModule::~CaloSimParametersESModule() {}

void CaloSimParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("caloSimulationParameters", desc);
}

CaloSimParametersESModule::ReturnType CaloSimParametersESModule::produce(const HcalParametersRcd& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::produce(const HcalParametersRcd& iRecord)";
#endif
  edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvToken_);

  auto ptp = std::make_unique<CaloSimulationParameters>();
  CaloSimParametersFromDD builder(fromDD4Hep_);
  builder.build(&(*cpv), *ptp);

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(CaloSimParametersESModule);
