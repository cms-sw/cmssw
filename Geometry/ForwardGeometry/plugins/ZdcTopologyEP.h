#ifndef GEOMETRY_HCALEVENTSETUP_ZDCTOPOLOGYEP_H
#define GEOMETRY_HCALEVENTSETUP_ZDCTOPOLOGYEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace edm {
  class ConfigurationDescriptions;
}

class ZdcTopologyEP : public edm::ESProducer {
public:
  ZdcTopologyEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<ZdcTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const HcalRecNumberingRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> m_hdcToken;
};
#endif
