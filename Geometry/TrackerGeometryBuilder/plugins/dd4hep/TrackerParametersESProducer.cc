#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
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

class TrackerParametersESProducer : public edm::ESProducer {
public:
  TrackerParametersESProducer(const edm::ParameterSet&);
  ~TrackerParametersESProducer(void) override;

  using ReturnType = std::unique_ptr<PTrackerParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const PTrackerParametersRcd&);
};

TrackerParametersESProducer::TrackerParametersESProducer(const edm::ParameterSet&) {
  edm::LogInfo("TRACKER") << "TrackerParametersESProducer::TrackerParametersESProducer";

  setWhatProduced(this);
}

TrackerParametersESProducer::~TrackerParametersESProducer() {}

void TrackerParametersESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("DD4hep_trackerParameters", desc);
}

TrackerParametersESProducer::ReturnType TrackerParametersESProducer::produce(const PTrackerParametersRcd& iRecord) {
  edm::LogInfo("TrackerParametersESProducer")
      << "TrackerParametersESProducer::produce(const PTrackerParametersRcd& iRecord)" << std::endl;
  edm::ESTransientHandle<cms::DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get(cpv);

  auto ptp = std::make_unique<PTrackerParameters>();
  TrackerParametersFromDD builder;
  builder.build(&(*cpv), *ptp);

  return ptp;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerParametersESProducer);
