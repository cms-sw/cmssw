
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
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
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  bool fromDD4Hep_;
};

CaloSimParametersESModule::CaloSimParametersESModule(const edm::ParameterSet& ps) {
  fromDD4Hep_ = ps.getParameter<bool>("fromDD4Hep");
  auto cc = setWhatProduced(this);
  if (fromDD4Hep_)
    cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  else
    cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::CaloSimParametersESModule called with dd4hep: "
                               << fromDD4Hep_;
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

  auto ptp = std::make_unique<CaloSimulationParameters>();
  CaloSimParametersFromDD builder;
  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "CaloSimParametersESModule::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp);
  }
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(CaloSimParametersESModule);
