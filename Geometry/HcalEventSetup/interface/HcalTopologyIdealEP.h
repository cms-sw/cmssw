#ifndef GEOMETRY_HCALEVENTSETUP_HCALTOPOLOGYIDEALEP_H
#define GEOMETRY_HCALEVENTSETUP_HCALTOPOLOGYIDEALEP_H 1

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HcalTopologyIdealEP : public edm::ESProducer {

public:
  HcalTopologyIdealEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  ReturnType produce(const HcalRecNumberingRecord&);

private:
  // ----------member data ---------------------------
  std::string m_restrictions;
  bool        m_mergePosition;
};
#endif
