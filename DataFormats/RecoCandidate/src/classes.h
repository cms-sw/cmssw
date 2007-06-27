#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/RecoCandidate/interface/FitResult.h"
#include "DataFormats/RecoCandidate/interface/CaloRecHitCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
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

    reco::RecoEcalCandidateIsolationMap v3;
    edm::Wrapper<reco::RecoEcalCandidateIsolationMap> w3;
    edm::helpers::Key<edm::RefProd<reco::RecoEcalCandidateCollection > > h3;


    edm::reftobase::Holder<reco::Candidate, reco::RecoEcalCandidateRef> rb1;
    edm::reftobase::Holder<reco::Candidate, reco::RecoChargedCandidateRef> rb2;
    edm::reftobase::Holder<CaloRecHit, HBHERecHitRef> rb4;
    edm::reftobase::Holder<CaloRecHit, HORecHitRef > rb5;
    edm::reftobase::Holder<CaloRecHit, HFRecHitRef> rb6;
    edm::reftobase::Holder<CaloRecHit, ZDCRecHitRef> rb7;
    edm::reftobase::Holder<CaloRecHit, EcalRecHitRef> rb8;
    edm::RefToBase<CaloRecHit> rbh3;

    reco::FitResultCollection fr1;
  }
}
