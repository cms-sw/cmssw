#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/GeometryObjects/interface/EcalSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalCommonData/interface/EcalSimParametersFromDD.h"

#include <memory>

//#define EDM_ML_DEBUG

class EcalSimParametersESModule : public edm::ESProducer {
public:
  EcalSimParametersESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<EcalSimulationParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  const bool fromDD4Hep_;
  const std::string name_;
};

EcalSimParametersESModule::EcalSimParametersESModule(const edm::ParameterSet& ps)
    : fromDD4Hep_(ps.getParameter<bool>("fromDD4Hep")), name_(ps.getParameter<std::string>("name")) {
  auto cc = setWhatProduced(this, name_);
  cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersESModule::EcalSimParametersESModule called with dd4hep: "
                               << fromDD4Hep_ << " for " << name_;
#endif
}

void EcalSimParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4Hep", false);
  desc.add<std::string>("name", "EcalHitsEB");
  descriptions.add("ecalSimulationParametersEB", desc);
}

EcalSimParametersESModule::ReturnType EcalSimParametersESModule::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersESModule::produce(const IdealGeometryRecord& iRecord)";
#endif

  auto ptp = std::make_unique<EcalSimulationParameters>();
  EcalSimParametersFromDD builder;
  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << "EcalSimParametersESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), name_, *ptp);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << "EcalSimParametersESModule::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), name_, *ptp);
  }
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(EcalSimParametersESModule);
