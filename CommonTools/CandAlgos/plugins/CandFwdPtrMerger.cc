/* \class CandFwdPtrMerger
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
#include "DataFormats/Common/interface/FwdPtr.h"

typedef UniqueMerger<std::vector<edm::FwdPtr<reco::Candidate>>> CandFwdPtrMerger;
typedef UniqueMerger<std::vector<edm::FwdPtr<pat::PackedCandidate>>> PackedCandidateFwdPtrMerger;
typedef UniqueMerger<std::vector<edm::FwdPtr<pat::PackedGenParticle>>> PackedGenParticleFwdPtrMerger;

DEFINE_FWK_MODULE(CandFwdPtrMerger);
DEFINE_FWK_MODULE(PackedCandidateFwdPtrMerger);
DEFINE_FWK_MODULE(PackedGenParticleFwdPtrMerger);
