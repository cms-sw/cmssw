#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

typedef SingleObjectSelector<
   std::vector<reco::RecoEcalCandidate>, 
   StringCutObjectSelector<reco::RecoEcalCandidate>,
   std::vector<reco::RecoEcalCandidate>
   > RecoEcalCandidateSelector;

DEFINE_FWK_MODULE( RecoEcalCandidateSelector );


typedef SingleObjectSelector<
  reco::RecoEcalCandidateCollection, 
  StringCutObjectSelector<reco::RecoEcalCandidate>,
  reco::RecoEcalCandidateRefVector
  > RecoEcalCandidateRefSelector;


//typedef SingleObjectSelector<
//   std::vector<reco::RecoEcalCandidate>, 
//   StringCutObjectSelector<reco::RecoEcalCandidate>,
//   std::vector<reco::RecoEcalCandidate>
//   > RecoEcalCandidateSelector;

DEFINE_FWK_MODULE( RecoEcalCandidateRefSelector );

