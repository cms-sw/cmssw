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
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Common/interface/FwdRef.h" 
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Ptr.h"

using namespace reco;

namespace {
  struct dictionary {
    CaloJetCollection o1;
    CaloJetRef r1;
    CaloJetFwdRef fwdr1;
    CaloJetFwdPtr fwdp1;
    CaloJetRefVector rr1;
    CaloJetFwdRefVector fwdrr1;
    CaloJetFwdPtrVector fwdpr1;
    CaloJetRefProd rrr1;
    edm::Wrapper<CaloJetCollection> w1;
    edm::Wrapper<CaloJetRefVector> wrv1;
    edm::Wrapper<CaloJetFwdRefVector> wfwdrv1;
    edm::Wrapper<CaloJetFwdPtrVector> wfwdpv1;
    edm::reftobase::Holder<reco::Candidate, reco::CaloJetRef> rtb1;
    JetTrackMatch<CaloJetCollection> jtm1;
    edm::AssociationMap<edm::OneToMany<std::vector<reco::CaloJet>,std::vector<reco::Track>,unsigned int> > amp1;
    edm::helpers::KeyVal<edm::RefProd<reco::CaloJetCollection>,edm::RefProd<std::vector<reco::Track> > > kv1;

    GenJetCollection o2;
    GenJetRef r2;
    GenJetFwdRef fwdr2;
    GenJetFwdPtr fwdp2;
    GenJetRefVector rr2;
    GenJetFwdRefVector fwdrr2;
    GenJetFwdPtrVector fwdpr2;
    GenJetRefProd rrr2;
    edm::Wrapper<GenJetCollection> w2;
    edm::Wrapper<GenJetRefVector> wrv2;
    edm::Wrapper<GenJetFwdRefVector> wfwdrv2;
    edm::Wrapper<GenJetFwdPtrVector> wfwdpv2;
    edm::reftobase::Holder<reco::Candidate, reco::GenJetRef> rtb2;
    JetTrackMatch<GenJetCollection> jtm2;

    BasicJetCollection o3;
    BasicJetRef r3;
    BasicJetFwdRef fwdr3;
    BasicJetFwdPtr fwdp3;
    BasicJetRefVector rr3;
    BasicJetFwdRefVector fwdrr3;
    BasicJetFwdPtrVector fwdrp3;
    BasicJetRefProd rrr3;
    edm::Wrapper<BasicJetCollection> w3;
    edm::Wrapper<BasicJetRefVector> wrv3;
    edm::Wrapper<BasicJetFwdRefVector> wfwdrv3;
    edm::Wrapper<BasicJetFwdPtrVector> wfwdpv3;
    edm::reftobase::Holder<reco::Candidate, reco::BasicJetRef> rtb3;
    JetTrackMatch<BasicJetCollection> jtm3;

    GenericJetCollection o4;
    GenericJetRef r4;
    GenericJetFwdRef fwdr4;
    GenericJetRefVector rr4;
    GenericJetFwdRefVector fwdrr4;
    GenericJetRefProd rrr4;
    edm::Wrapper<GenericJetCollection> w4;
    edm::Wrapper<GenericJetRefVector> wrv4;
    edm::Wrapper<GenericJetFwdRefVector> wfwdrv4;
    edm::reftobase::Holder<reco::Candidate, reco::GenericJetRef> rtb4;
    JetTrackMatch<GenericJetCollection> jtm4;

    PFJetCollection o5;
    PFJetRef r5;
    PFJetFwdRef fwdr5;
    PFJetFwdPtr fwdp5;
    PFJetRefVector rr5;
    PFJetFwdRefVector fwdrr5;
    PFJetFwdPtrVector fwdrp5;
    PFJetRefProd rrr5;
    edm::Wrapper<PFJetCollection> w5;
    edm::Wrapper<PFJetRefVector> wfwdrv5;
    edm::Wrapper<PFJetFwdRefVector> wrv5;
    edm::Wrapper<PFJetFwdPtrVector> wrp5;
    edm::reftobase::Holder<reco::Candidate, reco::PFJetRef> rtb5;
    JetTrackMatch<PFJetCollection> jtm5;

    TrackJetCollection o6;
    TrackJetRef r6;
    TrackJetFwdRef fwdr6;
    TrackJetFwdPtr fwdp6;
    TrackJetRefVector rr6;
    TrackJetFwdRefVector fwdrr6;
    TrackJetFwdPtrVector fwdrp6;
    TrackJetRefProd rrr6;
    edm::Wrapper<TrackJetCollection> w6;
    edm::Wrapper<TrackJetRefVector> wrv6;
    edm::Wrapper<TrackJetFwdRefVector> wfwdrv6;
    edm::Wrapper<TrackJetFwdPtrVector> wfwdrp6;
    edm::reftobase::Holder<reco::Candidate, reco::TrackJetRef> rtb6;
    JetTrackMatch<TrackJetCollection> jtm6;

    JPTJetCollection o7;
    JPTJetRef r7;
    JPTJetRefVector rr7;
    JPTJetRefProd rrr7;
    edm::Wrapper<JPTJetCollection> w7;
    edm::Wrapper<JPTJetRefVector> wrv7;
    edm::reftobase::Holder<reco::Candidate, reco::JPTJetRef> rtb7;
    JetTrackMatch<JPTJetCollection> jtm7;


    edm::reftobase::Holder<reco::Candidate,edm::RefToBase<reco::Jet> >  rtbb6;

    edm::Wrapper<JetFloatAssociation::Container>  jf_c_w;
    JetFloatAssociation::Container       jf_c;
    JetFloatAssociation::Ref             jf_r;
    JetFloatAssociation::RefProd         jf_rp;
    JetFloatAssociation::RefVector       jf_rv;

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

    // jet id stuff
    JetID jid;
    edm::Ref<std::vector<JetID> > rjid;
    edm::RefVector<std::vector<JetID> > rrjid;
    edm::RefProd<std::vector<JetID> > rrrjid;
    edm::Wrapper<std::vector<JetID> > wjid;
    edm::Wrapper<edm::RefVector<std::vector<JetID> > > wrvjid;
    edm::Wrapper<edm::ValueMap<JetID> > wvmjid;
    
    // castor jet id stuff
    CastorJetID cjid;
    edm::Ref<std::vector<CastorJetID> > crjid;
    edm::RefVector<std::vector<CastorJetID> > crrjid;
    edm::RefProd<std::vector<CastorJetID> > crrrjid;
    edm::Wrapper<std::vector<CastorJetID> > cwjid;
    edm::Wrapper<edm::RefVector<std::vector<CastorJetID> > > cwrvjid;
    edm::Wrapper<edm::ValueMap<CastorJetID> > cwvmjid;

    // Jet stuff
    edm::View<reco::Jet>  jv;
    edm::RefToBaseProd<reco::Jet> jrtbp;
    edm::RefToBase<reco::Jet> jrtb;
    

    // RefToBase Holders for Jets
    edm::RefToBase<reco::Jet>  rtbj;
    edm::reftobase::IndirectHolder<reco::Jet> ihj;
    edm::reftobase::Holder<reco::Jet, reco::CaloJetRef> hcj;
    edm::reftobase::Holder<reco::Jet, reco::JPTJetRef> hjptj;
    edm::reftobase::Holder<reco::Jet, reco::GenJetRef> hgj;
    edm::reftobase::Holder<reco::Jet, reco::PFJetRef> hpfj;
    edm::reftobase::Holder<reco::Jet, reco::BasicJetRef> hbj;
    edm::reftobase::Holder<reco::Jet, reco::TrackJetRef> htj;
    edm::reftobase::RefHolder<reco::CaloJetRef> rhcj;
    edm::reftobase::RefHolder<reco::JPTJetRef> rhjptj;
    edm::reftobase::RefHolder<reco::GenJetRef> rhgj;
    edm::reftobase::RefHolder<reco::PFJetRef> rhpfj;
    edm::reftobase::RefHolder<reco::BasicJetRef> rhbj;
    edm::reftobase::RefHolder<reco::TrackJetRef> rhtj;
    edm::RefToBaseVector<reco::Jet> jrtbv;
    edm::Wrapper<edm::RefToBaseVector<reco::Jet> > jrtbv_w;
    edm::reftobase::BaseVectorHolder<reco::Jet> * bvhj_p;    // pointer since it's pure virtual

    // Ptr stuff
    edm::Ptr<reco::Jet> ptrj;
    edm::PtrVector<reco::Jet> ptrvj;

    edm::Ptr<reco::CaloJet> ptrcj;
    edm::PtrVector<reco::CaloJet> ptrvcj;

    edm::Ptr<reco::JPTJet> ptrjptj;
    edm::PtrVector<reco::JPTJet> ptrvjptj;

    edm::Ptr<reco::PFJet> ptrpfj;
    edm::PtrVector<reco::PFJet> ptrvpfj;

    edm::Ptr<reco::BasicJet> ptrbj;
    edm::PtrVector<reco::BasicJet> ptrvbj;

    edm::Ptr<reco::GenJet> ptrgj;
    edm::PtrVector<reco::GenJet> ptrvgj;

    edm::Ptr<reco::TrackJet> ptrtj;
    edm::PtrVector<reco::TrackJet> ptrvtj;
    edm::Ptr<reco::Track> ptrt;
    std::vector<edm::Ptr<reco::Track> > vptrt;


    edm::Ptr<reco::JetID> ptrjid;
    edm::PtrVector<reco::JetID> ptrvjid;
    edm::Ptr<reco::CastorJetID> cptrjid;
    edm::PtrVector<reco::CastorJetID> cptrvjid;

    edm::Association<reco::GenJetCollection> a_gj;
    edm::Wrapper<edm::Association<reco::GenJetCollection> > w_a_gj;
    std::vector<reco::CaloJet::Specific> v_cj_s;
    std::vector<reco::JPTJet::Specific> v_jptj_s;
    std::vector<reco::PFJet::Specific> v_pj_s;

    reco::TrackExtrapolation xtrp;
    std::vector<reco::TrackExtrapolation> v_xtrp;
    edm::Ref<std::vector<reco::TrackExtrapolation> > r_xtrp;
    edm::RefVector<std::vector<reco::TrackExtrapolation> > rv_xtrp;
    edm::RefProd<std::vector<reco::TrackExtrapolation> > rp_xtrp;
    edm::Wrapper<reco::TrackExtrapolation> w_xtrp;
    edm::Wrapper<std::vector<reco::TrackExtrapolation> > wv_xtrp;
    edm::Wrapper<edm::Ref<std::vector<reco::TrackExtrapolation> > > wr_xtrp;
    edm::Wrapper<edm::RefVector<std::vector<reco::TrackExtrapolation> > > wrv_xtrp;
    edm::Wrapper<edm::RefProd<std::vector<reco::TrackExtrapolation> > > wrp_xtrp;

  };
}
#endif
