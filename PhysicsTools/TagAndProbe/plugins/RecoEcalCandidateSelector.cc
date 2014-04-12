#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

typedef SingleObjectSelector<
   std::vector<reco::RecoEcalCandidate>, 
   StringCutObjectSelector<reco::RecoEcalCandidate>,
   std::vector<reco::RecoEcalCandidate>
   > RecoEcalCandidateSelector;

DEFINE_FWK_MODULE( RecoEcalCandidateSelector );
