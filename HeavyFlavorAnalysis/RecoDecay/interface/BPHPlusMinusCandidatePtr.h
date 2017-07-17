#ifndef HeavyFlavorAnalysis_RecoDecay_BPHPlusMinusCandidatePtr_h
#define HeavyFlavorAnalysis_RecoDecay_BPHPlusMinusCandidatePtr_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHGenericPtr.h"
class BPHPlusMinusCandidate;
typedef BPHGenericPtr<      BPHPlusMinusCandidate>::type
                            BPHPlusMinusCandidatePtr;
typedef BPHGenericPtr<const BPHPlusMinusCandidate>::type
                            BPHPlusMinusConstCandPtr;

#endif
