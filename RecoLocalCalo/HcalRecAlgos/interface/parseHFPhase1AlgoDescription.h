#ifndef RecoLocalCalo_HcalRecAlgos_parseHFPhase1AlgoDescription_h
#define RecoLocalCalo_HcalRecAlgos_parseHFPhase1AlgoDescription_h

#include <memory>
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsHFPhase1Algo.h"

namespace edm {
    class ParameterSet;
}

//
// Factory function for creating objects of types
// inheriting from AbsHFPhase1Algo out of parameter sets.
//
// Update the implementation of this function if you need
// to add a new algorithm to HFPhase1Reconstructor.
//
std::unique_ptr<AbsHFPhase1Algo>
parseHFPhase1AlgoDescription(const edm::ParameterSet& ps);

#endif // RecoLocalCalo_HcalRecAlgos_parseHFPhase1AlgoDescription_h
