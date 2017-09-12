#ifndef RecoLocalCalo_HcalRecAlgos_parseHFPhase1AlgoDescription_h
#define RecoLocalCalo_HcalRecAlgos_parseHFPhase1AlgoDescription_h

#include <memory>
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsHFPhase1Algo.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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

//
// Parameter descriptions for "parseHFPhase1AlgoDescription".
// Keep implementation of this function is sync with
// "parseHFPhase1AlgoDescription".
//
edm::ParameterSetDescription fillDescriptionForParseHFPhase1AlgoDescription();

#endif // RecoLocalCalo_HcalRecAlgos_parseHFPhase1AlgoDescription_h
