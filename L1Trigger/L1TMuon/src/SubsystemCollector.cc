#include "L1Trigger/L1TMuon/interface/deprecate/SubsystemCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace L1TMuon;

SubsystemCollector::SubsystemCollector(const edm::ParameterSet& ps):
  _src(ps.getParameter<edm::InputTag>("src")) {
}
