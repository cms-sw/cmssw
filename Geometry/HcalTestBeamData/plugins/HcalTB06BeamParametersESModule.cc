#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParameters.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParametersFromDD.h"

#include <memory>

class HcalTB06ParametersESModule : public edm::ESProducer {
public:
  HcalTB06ParametersESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalTB06BeamParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  std::string name1_, name2_;
  bool fromDD4Hep_;
};

HcalTB06ParametersESModule::HcalTB06ParametersESModule(const edm::ParameterSet& ps) {
  name1_ = ps.getParameter<std::string>("name1");
  name2_ = ps.getParameter<std::string>("name2");
  fromDD4Hep_ = ps.getParameter<bool>("fromDD4Hep");
  auto cc = setWhatProduced(this);
  if (fromDD4Hep_)
    cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  else
    cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalTB06ParametersESModule::HcalTB06ParametersESModule called with dd4hep: "
                               << fromDD4Hep_ << " and names " << name1_ << ":" << name2_;
#endif
}

void HcalTB06ParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name1", "WireChamber");
  desc.add<std::string>("name2", "HcalTB06BeamHits");
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("hcalTB06BeamParameters", desc);
}

HcalTB06ParametersESModule::ReturnType HcalTB06ParametersESModule::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HCalGeom") << "HcalTB06ParametersESModule::produce(const IdealGeometryRecord& iRecord)";
#endif

  auto ptp = std::make_unique<HcalTB06BeamParameters>();
  HcalTB06BeamParametersFromDD builder;

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "HcalTB06ParametersESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp, name1_, name2_);
  } else {
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp, name1_, name2_);
  }
  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalTB06ParametersESModule);
