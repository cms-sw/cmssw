#ifndef RecoLocalCalo_HcalRecAlgos_parseHBHEPhase1AlgoDescription_h
#define RecoLocalCalo_HcalRecAlgos_parseHBHEPhase1AlgoDescription_h

#include <memory>
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsHBHEPhase1Algo.h"

namespace edm {
    class ParameterSet;
}

//
// Factory function for creating objects of types
// inheriting from AbsHBHEPhase1Algo out of parameter sets.
//
// Update the implementation of this function if you need
// to add a new algorithm to HBHEPhase1Reconstructor.
//
std::unique_ptr<AbsHBHEPhase1Algo>
parseHBHEPhase1AlgoDescription(const edm::ParameterSet& ps);

#endif // RecoLocalCalo_HcalRecAlgos_parseHBHEPhase1AlgoDescription_h
