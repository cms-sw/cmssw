#ifndef JetObjects1_classes_h
#define JetObjects1_classes_h

#include "DataFormats/JetReco/interface/BasicJet.h" 
#include "Rtypes.h" 

#include "DataFormats/JetReco/interface/BasicJetCollection.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h" 
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"
#include "DataFormats/JetReco/interface/JetTrackMatch.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/FwdRef.h" 
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace {
  struct dictionary1 {
    reco::CaloJetCollection o1;
    reco::CaloJetRef r1;
    reco::CaloJetFwdRef fwdr1;
    reco::CaloJetFwdPtr fwdp1;
    reco::CaloJetRefVector rr1;
    reco::CaloJetFwdRefVector fwdrr1;
    reco::CaloJetFwdPtrVector fwdpr1;
    reco::CaloJetRefProd rrr1;
    edm::Wrapper<reco::CaloJetCollection> w1;
    edm::Wrapper<reco::CaloJetRefVector> wrv1;
    edm::Wrapper<reco::CaloJetFwdRefVector> wfwdrv1;
    edm::Wrapper<reco::CaloJetFwdPtrVector> wfwdpv1;
    edm::reftobase::Holder<reco::Candidate, reco::CaloJetRef> rtb1;
    reco::JetTrackMatch<reco::CaloJetCollection> jtm1;
    edm::AssociationMap<edm::OneToMany<std::vector<reco::CaloJet>,std::vector<reco::Track>,unsigned int> > amp1;
    edm::helpers::Key<edm::RefProd<std::vector<reco::CaloJet> > > k1;
    edm::helpers::KeyVal<edm::RefProd<reco::CaloJetCollection>,edm::RefProd<std::vector<reco::Track> > > kv1;
    std::vector<edm::Ref<std::vector<reco::CaloJet> > > vrvr1;
    std::vector<reco::CaloJetRefVector> vrv1;    
    edm::Wrapper<std::vector<reco::CaloJetRefVector> > wfvrv1;

    reco::GenJetCollection o2;
    reco::GenJetRef r2;
    reco::GenJetFwdRef fwdr2;
    reco::GenJetFwdPtr fwdp2;
    reco::GenJetRefVector rr2;
    reco::GenJetFwdRefVector fwdrr2;
    reco::GenJetFwdPtrVector fwdpr2;
    reco::GenJetRefProd rrr2;
    edm::Wrapper<reco::GenJetCollection> w2;
    edm::Wrapper<reco::GenJetRefVector> wrv2;
    edm::Wrapper<reco::GenJetFwdRefVector> wfwdrv2;
    edm::Wrapper<reco::GenJetFwdPtrVector> wfwdpv2;
    edm::reftobase::Holder<reco::Candidate, reco::GenJetRef> rtb2;
    reco::JetTrackMatch<reco::GenJetCollection> jtm2;

    reco::BasicJetCollection o3;
    reco::BasicJetRef r3;
    reco::BasicJetFwdRef fwdr3;
    reco::BasicJetFwdPtr fwdp3;
    reco::BasicJetRefVector rr3;
    reco::BasicJetFwdRefVector fwdrr3;
    reco::BasicJetFwdPtrVector fwdrp3;
    reco::BasicJetRefProd rrr3;
    edm::Wrapper<reco::BasicJetCollection> w3;
    edm::Wrapper<reco::BasicJetRefVector> wrv3;
    edm::Wrapper<reco::BasicJetFwdRefVector> wfwdrv3;
    edm::Wrapper<reco::BasicJetFwdPtrVector> wfwdpv3;
    edm::reftobase::Holder<reco::Candidate, reco::BasicJetRef> rtb3;
    reco::JetTrackMatch<reco::BasicJetCollection> jtm3;

    reco::GenericJetCollection o4;
    reco::GenericJetRef r4;
    reco::GenericJetFwdRef fwdr4;
    reco::GenericJetRefVector rr4;
    reco::GenericJetFwdRefVector fwdrr4;
    reco::GenericJetRefProd rrr4;
    edm::Wrapper<reco::GenericJetCollection> w4;
    edm::Wrapper<reco::GenericJetRefVector> wrv4;
    edm::Wrapper<reco::GenericJetFwdRefVector> wfwdrv4;
    edm::reftobase::Holder<reco::Candidate, reco::GenericJetRef> rtb4;
    reco::JetTrackMatch<reco::GenericJetCollection> jtm4;

    reco::PFJetCollection o5;
    reco::PFJetRef r5;
    reco::PFJetFwdRef fwdr5;
    reco::PFJetFwdPtr fwdp5;
    reco::PFJetRefVector rr5;
    reco::PFJetFwdRefVector fwdrr5;
    reco::PFJetFwdPtrVector fwdrp5;
    reco::PFJetRefProd rrr5;
    edm::Wrapper<reco::PFJetCollection> w5;
    edm::Wrapper<reco::PFJetRefVector> wfwdrv5;
    edm::Wrapper<reco::PFJetFwdRefVector> wrv5;
    edm::Wrapper<reco::PFJetFwdPtrVector> wrp5;
    edm::reftobase::Holder<reco::Candidate, reco::PFJetRef> rtb5;
    reco::JetTrackMatch<reco::PFJetCollection> jtm5;
    std::vector<edm::Ref<std::vector<reco::PFJet> > > vrvr5;
    std::vector<reco::PFJetRefVector> vrv5;
    edm::Wrapper<std::vector<reco::PFJetRefVector> > wfvrv5;
    edm::reftobase::RefVectorHolder<reco::PFJetRefVector> rrpfr5;

    reco::JPTJetCollection o7;
    reco::JPTJetRef r7;
    reco::JPTJetRefVector rr7;
    reco::JPTJetRefProd rrr7;
    edm::Wrapper<reco::JPTJetCollection> w7;
    edm::Wrapper<reco::JPTJetRefVector> wrv7;
    edm::reftobase::Holder<reco::Candidate, reco::JPTJetRef> rtb7;
    reco::JetTrackMatch<reco::JPTJetCollection> jtm7;

    edm::reftobase::Holder<reco::Jet, reco::BasicJetRef> hbj;
    edm::reftobase::RefHolder<reco::BasicJetRef> rhbj;
  };
}
#endif
