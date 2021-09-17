#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB02Parameters.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB02ParametersFromDD.h"

#include <memory>

class HcalTB02ParametersESModule : public edm::ESProducer {
public:
  HcalTB02ParametersESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalTB02Parameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  std::string name_;
  bool fromDD4Hep_;
};

HcalTB02ParametersESModule::HcalTB02ParametersESModule(const edm::ParameterSet& ps) {
  name_ = ps.getParameter<std::string>("name");
  fromDD4Hep_ = ps.getParameter<bool>("fromDD4Hep");
  auto cc = setWhatProduced(this, name_);
  if (fromDD4Hep_)
    cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  else
    cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalTB02ParametersESModule::HcalTB02ParametersESModule called with dd4hep: "
                               << fromDD4Hep_ << " and name " << name_;
#endif
}

void HcalTB02ParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "EcalHitsEB");
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("hcalTB02XtalParameters", desc);
}

HcalTB02ParametersESModule::ReturnType HcalTB02ParametersESModule::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom") << "HcalTB02ParametersESModule::produce(const IdealGeometryRecord& iRecord)";
#endif

  auto ptp = std::make_unique<HcalTB02Parameters>(name_);
  HcalTB02ParametersFromDD builder;

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "HcalTB02ParametersESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp, name_);
  } else {
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp, name_);
  }
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalTB02ParametersESModule);
