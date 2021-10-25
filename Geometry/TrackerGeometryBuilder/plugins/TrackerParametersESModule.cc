#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

#include <memory>

namespace edm {
  class ConfigurationDescriptions;
}
class PTrackerParameters;
class PTrackerParametersRcd;

class TrackerParametersESModule : public edm::ESProducer {
public:
  TrackerParametersESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<PTrackerParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const PTrackerParametersRcd&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4Hep_;
  bool fromDD4Hep_;
};

TrackerParametersESModule::TrackerParametersESModule(const edm::ParameterSet& ps) {
  fromDD4Hep_ = ps.getParameter<bool>("fromDD4Hep");
  auto cc = setWhatProduced(this);
  if (fromDD4Hep_)
    cpvTokenDD4Hep_ = cc.consumesFrom<cms::DDCompactView, IdealGeometryRecord>(edm::ESInputTag());
  else
    cpvTokenDDD_ = cc.consumesFrom<DDCompactView, IdealGeometryRecord>(edm::ESInputTag());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TrackerGeom") << "TrackerParametersESModule::TrackerParametersESModule called with dd4hep: "
                                  << fromDD4Hep_;
#endif
}

void TrackerParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDD4Hep", false);
  descriptions.add("trackerParameters", desc);
}

TrackerParametersESModule::ReturnType TrackerParametersESModule::produce(const PTrackerParametersRcd& iRecord) {
  edm::LogInfo("TrackerGeom") << "TrackerParametersESModule::produce(const PTrackerParametersRcd& iRecord)";

  auto ptp = std::make_unique<PTrackerParameters>();
  TrackerParametersFromDD builder;

  if (fromDD4Hep_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("TrackerGeom") << "TrackerParametersESModule::produce try to access cms::DDCompactView";
#endif
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDD4Hep_);
    builder.build(&(*cpv), *ptp);
  } else {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("TrackerGeom") << "TrackerParametersESModule::produce try to access DDCompactView";
#endif
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvTokenDDD_);
    builder.build(&(*cpv), *ptp);
  }

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerParametersESModule);
