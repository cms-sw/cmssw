/* \class CandFwdPtrMerger
 * 
 * Producer of merged Candidate forward pointer collection 
 *
 * \author: Lauren Hay
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/Common/interface/FwdPtr.h"


typedef Merger<std::vector<edm::FwdPtr<reco::Candidate>>> CandFwdPtrMerger;
typedef Merger<std::vector<edm::FwdPtr<pat::PackedCandidate>>> PackedCandidateFwdPtrMerger;
typedef Merger<std::vector<edm::FwdPtr<pat::PackedGenParticle>>> PackedGenParticleFwdPtrMerger;


DEFINE_FWK_MODULE( CandFwdPtrMerger );
DEFINE_FWK_MODULE( PackedCandidateFwdPtrMerger );
DEFINE_FWK_MODULE( PackedGenParticleFwdPtrMerger );
