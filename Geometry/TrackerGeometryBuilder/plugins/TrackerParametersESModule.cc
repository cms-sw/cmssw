#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
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
  ~TrackerParametersESModule(void) override;

  using ReturnType = std::unique_ptr<PTrackerParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const PTrackerParametersRcd&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> geomToken_;
};

TrackerParametersESModule::TrackerParametersESModule(const edm::ParameterSet&) {
  edm::LogInfo("TRACKER") << "TrackerParametersESModule::TrackerParametersESModule";

  setWhatProduced(this).setConsumes(geomToken_);
}

TrackerParametersESModule::~TrackerParametersESModule() {}

void TrackerParametersESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("trackerParameters", desc);
}

TrackerParametersESModule::ReturnType TrackerParametersESModule::produce(const PTrackerParametersRcd& iRecord) {
  edm::LogInfo("TrackerParametersESModule")
      << "TrackerParametersESModule::produce(const PTrackerParametersRcd& iRecord)" << std::endl;
  edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(geomToken_);

  auto ptp = std::make_unique<PTrackerParameters>();
  TrackerParametersFromDD builder;
  builder.build(&(*cpv), *ptp);

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerParametersESModule);
