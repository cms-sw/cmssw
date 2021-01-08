/* \class CandPtrMerger
 * 
 * Producer of merged Candidate forward pointer collection 
 *
 * \author: Lauren Hay
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/UniqueMerger.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/Common/interface/Ptr.h"

typedef UniqueMerger<std::vector<edm::Ptr<reco::Candidate>>> CandPtrMerger;
typedef UniqueMerger<std::vector<edm::Ptr<pat::PackedCandidate>>> PackedCandidatePtrMerger;
typedef UniqueMerger<std::vector<edm::Ptr<pat::PackedGenParticle>>> PackedGenParticlePtrMerger;

DEFINE_FWK_MODULE(CandPtrMerger);
DEFINE_FWK_MODULE(PackedCandidatePtrMerger);
DEFINE_FWK_MODULE(PackedGenParticlePtrMerger);
