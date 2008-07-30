

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

typedef SingleObjectSelector<
   reco::RecoChargedCandidateCollection, 
   StringCutObjectSelector<reco::RecoChargedCandidate>,
   reco::RecoChargedCandidateRefVector
   > RecoChargedCandidateRefSelector;

DEFINE_FWK_MODULE( RecoChargedCandidateRefSelector );
