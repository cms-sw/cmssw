#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#define EDM_ML_DEBUG

class HGCalParametersESModule : public edm::ESProducer {
public:
  HGCalParametersESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HGCalParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const IdealGeometryRecord&);

private:
  std::string name_, name2_, namew_, namec_, namet_, namex_;
  bool fromDD4Hep_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
};

HGCalParametersESModule::HGCalParametersESModule(const edm::ParameterSet& iC) {
  name_ = iC.getParameter<std::string>("name");
  name2_ = iC.getParameter<std::string>("name2");
  namew_ = iC.getParameter<std::string>("nameW");
  namec_ = iC.getParameter<std::string>("nameC");
  namet_ = iC.getParameter<std::string>("nameT");
  namex_ = iC.getParameter<std::string>("nameX");
  fromDD4Hep_ = iC.getParameter<bool>("fromDD4Hep");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersESModule for " << name_ << ":" << namew_ << ":" << namec_ << ":"
                                << namet_ << " and fromDD4Hep flag " << fromDD4Hep_;
#endif
  auto cc = setWhatProduced(this, namex_);
  if (fromDD4Hep_)
    cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  else
    cpvTokenDDD_ = cc.consumes<DDCompactView>(edm::ESInputTag());
}

void HGCalParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "HGCalEESensitive");
  desc.add<std::string>("name2", "HGCalEE");
  desc.add<std::string>("nameW", "HGCalEEWafer");
  desc.add<std::string>("nameC", "HGCalEECell");
  desc.add<std::string>("nameT", "HGCal");
  desc.add<std::string>("nameX", "HGCalEESensitive");
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("hgcalEEParametersInitialize", desc);
}

HGCalParametersESModule::ReturnType HGCalParametersESModule::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersESModule::produce(const IdealGeometryRecord& iRecord)";
#endif
  auto ptp = std::make_unique<HGCalParameters>(name_);
  HGCalParametersFromDD builder;
  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalParametersESModule::Try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(cpv.product(), *ptp, name_, namew_, namec_, namet_, name2_);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalParametersESModule::Try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(cpv.product(), *ptp, name_, namew_, namec_, namet_);
  }
  return ptp;
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalParametersESModule);
