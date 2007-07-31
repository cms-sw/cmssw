#ifndef JetObjects_classes_h
#define JetObjects_classes_h

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/JetReco/interface/CaloJetfwd.h" 
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetfwd.h" 
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h" 
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetfwd.h" 
#include "DataFormats/JetReco/interface/GenericJet.h"
#include "DataFormats/JetReco/interface/GenericJetfwd.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetToFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetToTracksAssociation.h"
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

    JetToTracksAssociation::Container       j2t_c;
    JetToTracksAssociation::Object          j2t_o;
    JetToTracksAssociation::Ref             j2t_r;
    JetToTracksAssociation::RefProd         j2t_rp;
    JetToTracksAssociation::RefVector       j2t_rv;
    edm::Wrapper<JetToTracksAssociation::Container>  j2t_c_w;

    // RefToBase Holders for Jets
    edm::reftobase::Holder<reco::Jet, reco::CaloJetRef> rb_cj;
    edm::reftobase::Holder<reco::Jet, reco::GenJetRef>  rb_gj;
    edm::reftobase::Holder<reco::Jet, reco::PFJetRef>   rb_pfj;
  }
}
#endif
