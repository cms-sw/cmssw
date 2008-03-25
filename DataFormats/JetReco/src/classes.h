#ifndef JetObjects_classes_h
#define JetObjects_classes_h

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/GenericJet.h"
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetToFloatAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
 #include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"

using namespace reco;

namespace {
  namespace {
    CaloJetCollection o1;
    CaloJetRef r1;
    CaloJetRefVector rr1;
    CaloJetRefProd rrr1;
    edm::Wrapper<CaloJetCollection> w1;
    edm::reftobase::Holder<reco::Candidate, reco::CaloJetRef> rtb1;
    JetTrackMatch<CaloJetCollection> jtm1;
    edm::AssociationMap<edm::OneToMany<std::vector<reco::CaloJet>,std::vector<reco::Track>,unsigned int> > amp1;
    edm::helpers::KeyVal<edm::RefProd<reco::CaloJetCollection>,edm::RefProd<std::vector<reco::Track> > > kv1;

    GenJetCollection o2;
    GenJetRef r2;
    GenJetRefVector rr2;
    GenJetRefProd rrr2;
    edm::Wrapper<GenJetCollection> w2;
    edm::reftobase::Holder<reco::Candidate, reco::GenJetRef> rtb2;
    JetTrackMatch<GenJetCollection> jtm2;

    BasicJetCollection o3;
    BasicJetRef r3;
    BasicJetRefVector rr3;
    BasicJetRefProd rrr3;
    edm::Wrapper<BasicJetCollection> w3;
    edm::reftobase::Holder<reco::Candidate, reco::BasicJetRef> rtb3;
    JetTrackMatch<BasicJetCollection> jtm3;

    GenericJetCollection o4;
    GenericJetRef r4;
    GenericJetRefVector rr4;
    GenericJetRefProd rrr4;
    edm::Wrapper<GenericJetCollection> w4;
    edm::reftobase::Holder<reco::Candidate, reco::GenericJetRef> rtb4;
    JetTrackMatch<GenericJetCollection> jtm4;

    PFJetCollection o5;
    PFJetRef r5;
    PFJetRefVector rr5;
    PFJetRefProd rrr5;
    edm::Wrapper<PFJetCollection> w5;
    edm::reftobase::Holder<reco::Candidate, reco::PFJetRef> rtb5;
    JetTrackMatch<PFJetCollection> jtm5;

    JetToFloatAssociation::Container       j2f_c;
    JetToFloatAssociation::Object          j2f_o;
    JetToFloatAssociation::Objects         j2f_oo;
    edm::Wrapper<JetToFloatAssociation::Container>  j2f_c_w;

    edm::RefToBase<reco::Jet>  rbj1;
    edm::reftobase::IndirectHolder<reco::Jet> rbj3;
    edm::reftobase::Holder<reco::Jet, reco::CaloJetRef> rbj4;
    edm::reftobase::Holder<reco::Candidate,edm::RefToBase<reco::Jet> >  rtbb6;
    edm::RefToBaseProd<reco::Jet>   bp_jrtbp;

    edm::reftobase::Holder<reco::Candidate, reco::CaloJetRef> hccj1;
    edm::reftobase::RefHolder<reco::CaloJetRef> rhch1;
    edm::reftobase::VectorHolder<reco::Candidate, reco::CaloJetRefVector> vhccj1;
    edm::reftobase::RefVectorHolder<reco::CaloJetRefVector> rvhcj1;

    edm::reftobase::Holder<reco::Candidate, reco::GenJetRef> hcgj1;
    edm::reftobase::RefHolder<reco::GenJetRef> rhgj1;
    edm::reftobase::VectorHolder<reco::Candidate, reco::GenJetRefVector> vhcgj1;
    edm::reftobase::RefVectorHolder<reco::GenJetRefVector> rvhgj1;

    edm::reftobase::Holder<reco::Candidate, reco::BasicJetRef> hcbj1;
    edm::reftobase::RefHolder<reco::BasicJetRef> rhbj1;
    edm::reftobase::VectorHolder<reco::Candidate, reco::BasicJetRefVector> vhbj1;
    edm::reftobase::RefVectorHolder<reco::BasicJetRefVector> rvhbj1;

    edm::reftobase::Holder<reco::Candidate, reco::PFJetRef> hcpj1;
    edm::reftobase::RefHolder<reco::PFJetRef> rhpj1;
    edm::reftobase::VectorHolder<reco::Candidate, reco::PFJetRefVector> vhpj1;
    edm::reftobase::RefVectorHolder<reco::PFJetRefVector> rvhpj1;
  }
}
#endif
