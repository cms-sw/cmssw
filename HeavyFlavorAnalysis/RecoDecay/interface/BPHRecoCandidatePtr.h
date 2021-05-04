#ifndef HeavyFlavorAnalysis_RecoDecay_BPHRecoCandidatePtr_h
#define HeavyFlavorAnalysis_RecoDecay_BPHRecoCandidatePtr_h

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHGenericPtr.h"
class BPHRecoCandidate;
typedef BPHGenericPtr<BPHRecoCandidate> BPHRecoCandidateWrap;
typedef BPHGenericPtr<BPHRecoCandidate>::type BPHRecoCandidatePtr;
typedef BPHGenericPtr<const BPHRecoCandidate> BPHRecoConstCandWrap;
typedef BPHGenericPtr<const BPHRecoCandidate>::type BPHRecoConstCandPtr;

#endif
