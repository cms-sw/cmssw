#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace {
  namespace {
    reco::RecoChargedCandidateCollection v1;
    edm::Wrapper<reco::RecoChargedCandidateCollection> w1;
    edm::Ref<reco::RecoChargedCandidateCollection> r1;
    edm::RefProd<reco::RecoChargedCandidateCollection> rp1;
    edm::RefVector<reco::RecoChargedCandidateCollection> rv1;

    reco::RecoEcalCandidateCollection v2;
    edm::Wrapper<reco::RecoEcalCandidateCollection> w2;
    edm::Ref<reco::RecoEcalCandidateCollection> r2;
    edm::RefProd<reco::RecoEcalCandidateCollection> rp2;
    edm::RefVector<reco::RecoEcalCandidateCollection> rv2;

    std::map<unsigned int, float> m3;
    reco::RecoEcalCandidateIsolationMap v3;
    edm::Wrapper<reco::RecoEcalCandidateIsolationMap> w3;
    edm::helpers::Key<edm::RefProd<reco::RecoEcalCandidateCollection > > h3;


    edm::reftobase::Holder<reco::Candidate, reco::RecoEcalCandidateRef> rb1;
    edm::reftobase::Holder<reco::Candidate, reco::RecoChargedCandidateRef> rb2;
  }
}
