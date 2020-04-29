#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDParameters.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstantsBuild.h"

#include <memory>

//#define EDM_ML_DEBUG

class MuonDDDConstantsESModule : public edm::ESProducer {
public:
  MuonDDDConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MuonDDDParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  bool fromDD4Hep_;
};

MuonDDDConstantsESModule::MuonDDDConstantsESModule(const edm::ParameterSet& ps) {
  fromDD4Hep_ = ps.getParameter<bool>("fromDD4Hep");
  auto cc = setWhatProduced(this);
  cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "MuonDDDConstantsESModule::MuonDDDConstantsESModule called with dd4hep: "
                               << fromDD4Hep_;
#endif
}

void MuonDDDConstantsESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("muonDDDConstants", desc);
}

MuonDDDConstantsESModule::ReturnType MuonDDDConstantsESModule::produce(const IdealGeometryRecord& iRecord) {
  edm::LogInfo("Geometry") << "MuonDDDConstantsESModule::produce(const IdealGeometryRecord& iRecord)";

  auto ptp = std::make_unique<MuonDDDParameters>();
  MuonDDDConstantsBuild builder;

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Geometry") << "MuonDDDConstantsESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("Geometry") << "MuonDDDConstantsESModule::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp);
  }

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonDDDConstantsESModule);
