#ifndef RecoLocalCalo_HcalRecAlgos_parseHBHEPhase1AlgoDescription_h
#define RecoLocalCalo_HcalRecAlgos_parseHBHEPhase1AlgoDescription_h

#include <memory>
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsHBHEPhase1Algo.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// Factory function for creating objects of types
// inheriting from AbsHBHEPhase1Algo out of parameter sets.
//
// Update the implementation of this function if you need
// to add a new algorithm to HBHEPhase1Reconstructor.
//
std::unique_ptr<AbsHBHEPhase1Algo> parseHBHEPhase1AlgoDescription(const edm::ParameterSet& ps,
                                                                  edm::ConsumesCollector iC);

//
// Parameter descriptions for "parseHBHEPhase1AlgoDescription".
// Keep implementation of this function is sync with
// "parseHBHEPhase1AlgoDescription".
//
edm::ParameterSetDescription fillDescriptionForParseHBHEPhase1Algo();

#endif  // RecoLocalCalo_HcalRecAlgos_parseHBHEPhase1AlgoDescription_h
