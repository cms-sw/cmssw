#ifndef JetObjects_classes_h
#define JetObjects_classes_h

#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h" 
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PtEtaPhiM4D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h" 
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetToFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
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

    edm::reftobase::Holder<reco::Candidate,edm::RefToBase<reco::Jet> >  rtbb6;

    edm::Wrapper<JetFloatAssociation::Container>  jf_c_w;
    JetFloatAssociation::Container       jf_c;
    JetFloatAssociation::Ref             jf_r;
    JetFloatAssociation::RefProd         jf_rp;
    JetFloatAssociation::RefVector       jf_rv;

    edm::Wrapper<JetToFloatAssociation::Container>  j2f_c_w;
    JetToFloatAssociation::Container       j2f_c;
    JetToFloatAssociation::Ref             j2f_r;
    JetToFloatAssociation::RefProd         j2f_rp;
    JetToFloatAssociation::RefVector       j2f_rv;

    edm::Wrapper<JetTracksAssociation::Container>  jt_c_w;
    JetTracksAssociation::Container       jt_c;
    JetTracksAssociation::Ref             jt_r;
    JetTracksAssociation::RefProd         jt_rp;
    JetTracksAssociation::RefVector       jt_rv;

    edm::Wrapper<JetExtendedAssociation::Container>  jea_c_w;
    JetExtendedAssociation::Container       jea_c;
    JetExtendedAssociation::Ref             jea_r;
    JetExtendedAssociation::RefProd         jea_rp;
    JetExtendedAssociation::RefVector       jea_rv;

    // Jet stuff
    edm::View<reco::Jet>  jv;
    edm::RefToBaseProd<reco::Jet> jrtbp;
    edm::RefToBase<reco::Jet> jrtb;
    

    // RefToBase Holders for Jets
    edm::RefToBase<reco::Jet>  rtbj;
    edm::reftobase::IndirectHolder<reco::Jet> ihj;
    edm::reftobase::Holder<reco::Jet, reco::CaloJetRef> hcj;
    edm::reftobase::Holder<reco::Jet, reco::GenJetRef> hgj;
    edm::reftobase::Holder<reco::Jet, reco::PFJetRef> hpfj;
    edm::reftobase::RefHolder<reco::CaloJetRef> rhcj;
    edm::reftobase::RefHolder<reco::GenJetRef> rhgj;
    edm::reftobase::RefHolder<reco::PFJetRef> rhpfj;
  }
}
#endif
