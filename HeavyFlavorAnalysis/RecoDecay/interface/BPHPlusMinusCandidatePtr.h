#ifndef BPHPlusMinusCandidatePtr_H
#define BPHPlusMinusCandidatePtr_H

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHGenericPtr.h"
class BPHPlusMinusCandidate;
typedef BPHGenericPtr<      BPHPlusMinusCandidate>::type
                            BPHPlusMinusCandidatePtr;
typedef BPHGenericPtr<const BPHPlusMinusCandidate>::type
                            BPHPlusMinusConstCandPtr;

#endif
