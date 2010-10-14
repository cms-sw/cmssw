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

namespace {
  struct dictionary {
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
    edm::helpers::KeyVal<edm::RefProd<reco::CaloJetCollection>,edm::RefProd<std::vector<reco::Track> > > kv1;

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

    reco::TrackJetCollection o6;
    reco::TrackJetRef r6;
    reco::TrackJetFwdRef fwdr6;
    reco::TrackJetFwdPtr fwdp6;
    reco::TrackJetRefVector rr6;
    reco::TrackJetFwdRefVector fwdrr6;
    reco::TrackJetFwdPtrVector fwdrp6;
    reco::TrackJetRefProd rrr6;
    edm::Wrapper<reco::TrackJetCollection> w6;
    edm::Wrapper<reco::TrackJetRefVector> wrv6;
    edm::Wrapper<reco::TrackJetFwdRefVector> wfwdrv6;
    edm::Wrapper<reco::TrackJetFwdPtrVector> wfwdrp6;
    edm::reftobase::Holder<reco::Candidate, reco::TrackJetRef> rtb6;
    reco::JetTrackMatch<reco::TrackJetCollection> jtm6;

    reco::JPTJetCollection o7;
    reco::JPTJetRef r7;
    reco::JPTJetRefVector rr7;
    reco::JPTJetRefProd rrr7;
    edm::Wrapper<reco::JPTJetCollection> w7;
    edm::Wrapper<reco::JPTJetRefVector> wrv7;
    edm::reftobase::Holder<reco::Candidate, reco::JPTJetRef> rtb7;
    reco::JetTrackMatch<reco::JPTJetCollection> jtm7;


    edm::reftobase::Holder<reco::Candidate,edm::RefToBase<reco::Jet> >  rtbb6;

    edm::Wrapper<reco::JetFloatAssociation::Container>  jf_c_w;
    reco::JetFloatAssociation::Container       jf_c;
    reco::JetFloatAssociation::Ref             jf_r;
    reco::JetFloatAssociation::RefProd         jf_rp;
    reco::JetFloatAssociation::RefVector       jf_rv;

    edm::Wrapper<reco::JetTracksAssociation::Container>  jt_c_w;
    reco::JetTracksAssociation::Container       jt_c;
    reco::JetTracksAssociation::Ref             jt_r;
    reco::JetTracksAssociation::RefProd         jt_rp;
    reco::JetTracksAssociation::RefVector       jt_rv;

    edm::Wrapper<reco::JetExtendedAssociation::Container>  jea_c_w;
    reco::JetExtendedAssociation::Container       jea_c;
    reco::JetExtendedAssociation::Ref             jea_r;
    reco::JetExtendedAssociation::RefProd         jea_rp;
    reco::JetExtendedAssociation::RefVector       jea_rv;

    // jet id stuff
    reco::JetID jid;
    edm::Ref<std::vector<reco::JetID> > rjid;
    edm::RefVector<std::vector<reco::JetID> > rrjid;
    edm::RefProd<std::vector<reco::JetID> > rrrjid;
    edm::Wrapper<std::vector<reco::JetID> > wjid;
    edm::Wrapper<edm::RefVector<std::vector<reco::JetID> > > wrvjid;
    edm::Wrapper<edm::ValueMap<reco::JetID> > wvmjid;
    
    // castor jet id stuff
    reco::CastorJetID cjid;
    edm::Ref<std::vector<reco::CastorJetID> > crjid;
    edm::RefVector<std::vector<reco::CastorJetID> > crrjid;
    edm::RefProd<std::vector<reco::CastorJetID> > crrrjid;
    edm::Wrapper<std::vector<reco::CastorJetID> > cwjid;
    edm::Wrapper<edm::RefVector<std::vector<reco::CastorJetID> > > cwrvjid;
    edm::Wrapper<edm::ValueMap<reco::CastorJetID> > cwvmjid;

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
