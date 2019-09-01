#include "RecoLocalCalo/HcalRecAlgos/interface/parsePlan1RechitCombiner.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Concrete "Plan 1" rechit combination algorithm headers
#include "RecoLocalCalo/HcalRecAlgos/interface/SimplePlan1RechitCombiner.h"

std::unique_ptr<AbsPlan1RechitCombiner> parsePlan1RechitCombiner(const edm::ParameterSet& ps) {
  std::unique_ptr<AbsPlan1RechitCombiner> algo;

  const std::string& className = ps.getParameter<std::string>("Class");

  if (className == "SimplePlan1RechitCombiner")
    algo = std::unique_ptr<AbsPlan1RechitCombiner>(new SimplePlan1RechitCombiner());

  return algo;
}
