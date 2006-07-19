#ifndef JetObjects_classes_h
#define JetObjects_classes_h

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Common/interface/RefToBase.h"

using namespace reco;

namespace {
  namespace {
    CaloJetCollection o1;
    CaloJetRef r1;
    CaloJetRefVector rr1;
    CaloJetRefProd rrr1;
    edm::Wrapper<CaloJetCollection> w1;
    edm::reftobase::Holder<reco::Candidate, reco::CaloJetRef> rtb1;

    GenJetCollection o2;
    GenJetRef r2;
    GenJetRefVector rr2;
    GenJetRefProd rrr2;
    edm::Wrapper<GenJetCollection> w2;
    edm::reftobase::Holder<reco::Candidate, reco::GenJetRef> rtb2;

    BasicJetCollection o3;
    BasicJetRef r3;
    BasicJetRefVector rr3;
    BasicJetRefProd rrr3;
    edm::Wrapper<BasicJetCollection> w3;
    edm::reftobase::Holder<reco::Candidate, reco::BasicJetRef> rtb3;
  }
}
#endif
