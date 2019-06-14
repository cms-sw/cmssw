#ifndef RecoLocalCalo_HcalRecAlgos_parsePlan1RechitCombiner_h_
#define RecoLocalCalo_HcalRecAlgos_parsePlan1RechitCombiner_h_

#include <memory>
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsPlan1RechitCombiner.h"

namespace edm {
  class ParameterSet;
}

//
// Factory function for creating objects of types inheriting from
// AbsPlan1RechitCombiner out of parameter sets
//
std::unique_ptr<AbsPlan1RechitCombiner> parsePlan1RechitCombiner(const edm::ParameterSet& ps);

#endif  // RecoLocalCalo_HcalRecAlgos_parsePlan1RechitCombiner_h_
