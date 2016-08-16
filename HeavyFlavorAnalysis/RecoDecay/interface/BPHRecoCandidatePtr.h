#ifndef BPHRecoCandidatePtr_H
#define BPHRecoCandidatePtr_H

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHGenericPtr.h"
class BPHRecoCandidate;
typedef BPHGenericPtr<      BPHRecoCandidate>::type BPHRecoCandidatePtr;
typedef BPHGenericPtr<const BPHRecoCandidate>::type BPHRecoConstCandPtr;

#endif
