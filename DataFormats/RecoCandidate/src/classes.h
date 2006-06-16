#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloJetCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    reco::RecoCaloJetCandidateCollection c1;
    edm::Wrapper<reco::RecoCaloJetCandidateCollection> w1;
    reco::RecoCaloJetCandidateRef r1;
    reco::RecoCaloJetCandidateRefVector rv1;
    reco::RecoCaloJetCandidateRefProd rp1;
  }
}
