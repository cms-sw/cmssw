#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "EgammaHLTFilteredObjProducer.h"

template <>
void EgammaHLTFilteredObjProducer<reco::RecoEcalCandidateCollection>::addObj(
    const reco::RecoEcalCandidateRef& cand, reco::RecoEcalCandidateCollection& output) {
  output.push_back(*cand);
}

using EgammaHLTFilteredEcalCandProducer = EgammaHLTFilteredObjProducer<reco::RecoEcalCandidateCollection>;
DEFINE_FWK_MODULE(EgammaHLTFilteredEcalCandProducer);

template <>
void EgammaHLTFilteredObjProducer<std::vector<edm::Ptr<reco::Candidate> > >::addObj(
    const reco::RecoEcalCandidateRef& cand, std::vector<edm::Ptr<reco::Candidate> >& output) {
  output.push_back(edm::refToPtr(cand));
}

using EgammaHLTFilteredEcalCandPtrProducer = EgammaHLTFilteredObjProducer<std::vector<edm::Ptr<reco::Candidate> > >;
DEFINE_FWK_MODULE(EgammaHLTFilteredEcalCandPtrProducer);
