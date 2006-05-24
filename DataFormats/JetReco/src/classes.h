#ifndef JetObjects_classes_h
#define JetObjects_classes_h

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/Wrapper.h"
namespace {
  namespace {
    CaloJetCollection o1;
    CaloJetRef r1;
    CaloJetRefs rr1;
    edm::RefVector<std::vector<CaloJet> > rr11;
    CaloJetsRef rrr1;
    edm::Wrapper<CaloJetCollection> w1;
    GenJetCollection o2;
    GenJetRef r2;
    GenJetRefs rr2;
    GenJetsRef rrr2;
    edm::Wrapper<GenJetCollection> w2;
  }
}
#endif
