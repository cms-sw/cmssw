
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DPGAnalysis/Skims/interface/MatchedProbeMaker.h"

typedef MatchedProbeMaker< reco::RecoChargedCandidate > RecoChargedCandidateMatchedProbeMaker;

DEFINE_FWK_MODULE( RecoChargedCandidateMatchedProbeMaker );
