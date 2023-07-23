#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"

#include <memory>

namespace edm {
  class ConfigurationDescriptions;
}
class PTrackerAdditionalParametersPerDet;
class PTrackerAdditionalParametersPerDetRcd;

class TrackerAdditionalParametersPerDetESModule : public edm::ESProducer {
public:
  TrackerAdditionalParametersPerDetESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<PTrackerAdditionalParametersPerDet>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const PTrackerAdditionalParametersPerDetRcd&);

private:
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geometricDetToken_;
};

TrackerAdditionalParametersPerDetESModule::TrackerAdditionalParametersPerDetESModule(const edm::ParameterSet& ps) {
  auto cc = setWhatProduced(this);
  geometricDetToken_ = cc.consumesFrom<GeometricDet, IdealGeometryRecord>(edm::ESInputTag());
}

void TrackerAdditionalParametersPerDetESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("TrackerAdditionalParametersPerDet", desc);
}

TrackerAdditionalParametersPerDetESModule::ReturnType TrackerAdditionalParametersPerDetESModule::produce(
    const PTrackerAdditionalParametersPerDetRcd& iRecord) {
  edm::LogInfo("TrackerAdditionalParametersPerDet")
      << "TrackerAdditionalParametersPerDetESModule::produce(const PTrackerAdditionalParametersPerDetRcd& iRecord)";

  auto ptitp = std::make_unique<PTrackerAdditionalParametersPerDet>();

  edm::ESTransientHandle<GeometricDet> gd = iRecord.getTransientHandle(geometricDetToken_);

  std::vector<const GeometricDet*> comp;
  gd->deepComponents(comp);

  for (auto& i : comp) {
    ptitp->setGeographicalId(i->geographicalId());
  }

  return ptitp;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerAdditionalParametersPerDetESModule);
