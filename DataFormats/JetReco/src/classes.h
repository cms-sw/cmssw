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

#include "DataFormats/JetReco/interface/BasicJetCollection.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h" 
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/PFClusterJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/JetReco/interface/CastorJetID.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"

#include "DataFormats/JetReco/interface/PattRecoPeak.h"
#include "DataFormats/JetReco/interface/PattRecoNode.h"
#include "DataFormats/JetReco/interface/PattRecoTree.h"
#include "DataFormats/JetReco/interface/FFTJetProducerSummary.h"
#include "DataFormats/JetReco/interface/FFTCaloJetCollection.h" 
#include "DataFormats/JetReco/interface/FFTJPTJetCollection.h"
#include "DataFormats/JetReco/interface/FFTGenJetCollection.h"
#include "DataFormats/JetReco/interface/FFTPFJetCollection.h"
#include "DataFormats/JetReco/interface/FFTTrackJetCollection.h"
#include "DataFormats/JetReco/interface/FFTBasicJetCollection.h"
#include "DataFormats/JetReco/interface/FFTJetPileupSummary.h"
#include "DataFormats/JetReco/interface/DiscretizedEnergyFlow.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"

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

    reco::PFClusterJetCollection o8;
    reco::PFClusterJetRef r8;
    reco::PFClusterJetFwdRef fwdr8;
    reco::PFClusterJetFwdPtr fwdp8;
    reco::PFClusterJetRefVector rr8;
    reco::PFClusterJetFwdRefVector fwdrr8;
    reco::PFClusterJetFwdPtrVector fwdrp8;
    reco::PFClusterJetRefProd rrr8;
    edm::Wrapper<reco::PFClusterJetCollection> w8;
    edm::Wrapper<reco::PFClusterJetRefVector> wrv8;
    edm::Wrapper<reco::PFClusterJetFwdRefVector> wfwdrv8;
    edm::Wrapper<reco::PFClusterJetFwdPtrVector> wfwdrp8;
    edm::reftobase::Holder<reco::Candidate, reco::PFClusterJetRef> rtb8;
    reco::JetTrackMatch<reco::PFClusterJetCollection> jtm8;

    StoredPileupJetIdentifier spujetid;
    std::vector<StoredPileupJetIdentifier> spujetidvec;
    edm::ValueMap<StoredPileupJetIdentifier> spujetidvmap;
    edm::Wrapper<edm::ValueMap<StoredPileupJetIdentifier> > spujetidvmapw;

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
    std::pair<edm::RefToBase<reco::Jet>,reco::JetExtendedAssociation::JetExtendedData> jea_pair;

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
    std::pair<edm::RefToBase<reco::Jet>,edm::RefVector<std::vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<std::vector<reco::Track>,reco::Track> > > pjrtbt;

    // RefToBase Holders for Jets
    edm::RefToBase<reco::Jet>  rtbj;
    edm::reftobase::IndirectHolder<reco::Jet> ihj;
    edm::reftobase::IndirectVectorHolder<reco::Jet> ihvj;
    edm::reftobase::Holder<reco::Jet, reco::CaloJetRef> hcj;
    edm::reftobase::Holder<reco::Jet, reco::JPTJetRef> hjptj;
    edm::reftobase::Holder<reco::Jet, reco::GenJetRef> hgj;
    edm::reftobase::Holder<reco::Jet, reco::PFJetRef> hpfj;
    edm::reftobase::Holder<reco::Jet, reco::BasicJetRef> hbj;
    edm::reftobase::Holder<reco::Jet, reco::TrackJetRef> htj;
    edm::reftobase::Holder<reco::Jet, reco::PFClusterJetRef> hpfcj;
    edm::reftobase::RefHolder<reco::CaloJetRef> rhcj;
    edm::reftobase::RefHolder<reco::JPTJetRef> rhjptj;
    edm::reftobase::RefHolder<reco::GenJetRef> rhgj;
    edm::reftobase::RefHolder<reco::PFJetRef> rhpfj;
    edm::reftobase::RefHolder<reco::BasicJetRef> rhbj;
    edm::reftobase::RefHolder<reco::TrackJetRef> rhtj;
    edm::reftobase::RefHolder<reco::PFClusterJetRef> rhpfcj;
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

    edm::Ptr<reco::PFClusterJet> ptrpfcj;
    edm::PtrVector<reco::PFClusterJet> ptrvpfcj;
    edm::Ptr<reco::PFCluster> ptrpfc;
    std::vector<edm::Ptr<reco::PFCluster> > vptrpfc;



    edm::Ptr<reco::JetID> ptrjid;
    edm::PtrVector<reco::JetID> ptrvjid;
    edm::Ptr<reco::CastorJetID> cptrjid;
    edm::PtrVector<reco::CastorJetID> cptrvjid;

    edm::Association<reco::GenJetCollection> a_gj;
    edm::Wrapper<edm::Association<reco::GenJetCollection> > w_a_gj;

    edm::Association<reco::PFJetCollection> a_pfj;
    edm::Wrapper<edm::Association<reco::PFJetCollection> > w_a_pfj;

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

    // FFTJet interface
    reco::FFTJet<float> fftjet_float;
    reco::FFTJet<double> fftjet_double;
    reco::PattRecoPeak<float> pattRecoPeak_float;
    reco::PattRecoPeak<double> pattRecoPeak_double;
    reco::PattRecoNode<reco::PattRecoPeak<float> > pattRecoNode_Peak_float;
    reco::PattRecoNode<reco::PattRecoPeak<double> > pattRecoNode_Peak_double;
    std::vector<reco::PattRecoNode<reco::PattRecoPeak<float> > > v_pattRecoNode_Peak_float;
    std::vector<reco::PattRecoNode<reco::PattRecoPeak<double> > > v_pattRecoNode_Peak_double;
    reco::PattRecoTree<float,reco::PattRecoPeak<float> > pattRecoTree_Peak_float;
    reco::PattRecoTree<double,reco::PattRecoPeak<double> > pattRecoTree_Peak_double;
    edm::Wrapper<reco::PattRecoTree<float,reco::PattRecoPeak<float> > > wr_pattRecoTree_Peak_double;
    edm::Wrapper<reco::PattRecoTree<double,reco::PattRecoPeak<double> > > wr_pattRecoTree_Peak_float;

    reco::FFTJetProducerSummary fftjet_smry;
    edm::Wrapper<reco::FFTJetProducerSummary> wr_fftjet_smry;

    reco::FFTGenJet jet_fft_1;
    reco::FFTGenJetCollection o2_fft_1;
    reco::FFTGenJetRef r2_fft_1;
    reco::FFTGenJetFwdRef fwdr2_fft_1;
    reco::FFTGenJetFwdPtr fwdp2_fft_1;
    reco::FFTGenJetRefVector rr2_fft_1;
    reco::FFTGenJetFwdRefVector fwdrr2_fft_1;
    reco::FFTGenJetFwdPtrVector fwdpr2_fft_1;
    reco::FFTGenJetRefProd rrr2_fft_1;
    edm::Wrapper<reco::FFTGenJetCollection> w2_fft_1;
    edm::Wrapper<reco::FFTGenJetRefVector> wrv2_fft_1;
    edm::Wrapper<reco::FFTGenJetFwdRefVector> wfwdrv2_fft_1;
    edm::Wrapper<reco::FFTGenJetFwdPtrVector> wfwdpv2_fft_1;
    edm::reftobase::Holder<reco::Candidate, reco::FFTGenJetRef> rtb2_fft_1;
    reco::JetTrackMatch<reco::FFTGenJetCollection> jtm2_fft_1;
    edm::reftobase::Holder<reco::Jet, reco::FFTGenJetRef> hgj_fft_1;
    edm::reftobase::RefHolder<reco::FFTGenJetRef> rhgj_fft_1;
    edm::Ptr<reco::FFTGenJet> ptrgj_fft_1;
    edm::PtrVector<reco::FFTGenJet> ptrvgj_fft_1;
    edm::Association<reco::FFTGenJetCollection> a_gj_fft_1;
    edm::Wrapper<edm::Association<reco::FFTGenJetCollection> > w_a_gj_fft_1;

    reco::FFTCaloJet jet_fft_2;
    reco::FFTCaloJetCollection o2_fft_2;
    reco::FFTCaloJetRef r2_fft_2;
    reco::FFTCaloJetFwdRef fwdr2_fft_2;
    reco::FFTCaloJetFwdPtr fwdp2_fft_2;
    reco::FFTCaloJetRefVector rr2_fft_2;
    reco::FFTCaloJetFwdRefVector fwdrr2_fft_2;
    reco::FFTCaloJetFwdPtrVector fwdpr2_fft_2;
    reco::FFTCaloJetRefProd rrr2_fft_2;
    edm::Wrapper<reco::FFTCaloJetCollection> w2_fft_2;
    edm::Wrapper<reco::FFTCaloJetRefVector> wrv2_fft_2;
    edm::Wrapper<reco::FFTCaloJetFwdRefVector> wfwdrv2_fft_2;
    edm::Wrapper<reco::FFTCaloJetFwdPtrVector> wfwdpv2_fft_2;
    edm::reftobase::Holder<reco::Candidate, reco::FFTCaloJetRef> rtb2_fft_2;
    reco::JetTrackMatch<reco::FFTCaloJetCollection> jtm2_fft_2;
    edm::reftobase::Holder<reco::Jet, reco::FFTCaloJetRef> hgj_fft_2;
    edm::reftobase::RefHolder<reco::FFTCaloJetRef> rhgj_fft_2;
    edm::Ptr<reco::FFTCaloJet> ptrgj_fft_2;
    edm::PtrVector<reco::FFTCaloJet> ptrvgj_fft_2;
    edm::Association<reco::FFTCaloJetCollection> a_gj_fft_2;
    edm::Wrapper<edm::Association<reco::FFTCaloJetCollection> > w_a_gj_fft_2;

    reco::FFTBasicJet jet_fft_3;
    reco::FFTBasicJetCollection o2_fft_3;
    reco::FFTBasicJetRef r2_fft_3;
    reco::FFTBasicJetFwdRef fwdr2_fft_3;
    reco::FFTBasicJetFwdPtr fwdp2_fft_3;
    reco::FFTBasicJetRefVector rr2_fft_3;
    reco::FFTBasicJetFwdRefVector fwdrr2_fft_3;
    reco::FFTBasicJetFwdPtrVector fwdpr2_fft_3;
    reco::FFTBasicJetRefProd rrr2_fft_3;
    edm::Wrapper<reco::FFTBasicJetCollection> w2_fft_3;
    edm::Wrapper<reco::FFTBasicJetRefVector> wrv2_fft_3;
    edm::Wrapper<reco::FFTBasicJetFwdRefVector> wfwdrv2_fft_3;
    edm::Wrapper<reco::FFTBasicJetFwdPtrVector> wfwdpv2_fft_3;
    edm::reftobase::Holder<reco::Candidate, reco::FFTBasicJetRef> rtb2_fft_3;
    reco::JetTrackMatch<reco::FFTBasicJetCollection> jtm2_fft_3;
    edm::reftobase::Holder<reco::Jet, reco::FFTBasicJetRef> hgj_fft_3;
    edm::reftobase::RefHolder<reco::FFTBasicJetRef> rhgj_fft_3;
    edm::Ptr<reco::FFTBasicJet> ptrgj_fft_3;
    edm::PtrVector<reco::FFTBasicJet> ptrvgj_fft_3;
    edm::Association<reco::FFTBasicJetCollection> a_gj_fft_3;
    edm::Wrapper<edm::Association<reco::FFTBasicJetCollection> > w_a_gj_fft_3;

    reco::FFTPFJet jet_fft_4;
    reco::FFTPFJetCollection o2_fft_4;
    reco::FFTPFJetRef r2_fft_4;
    reco::FFTPFJetFwdRef fwdr2_fft_4;
    reco::FFTPFJetFwdPtr fwdp2_fft_4;
    reco::FFTPFJetRefVector rr2_fft_4;
    reco::FFTPFJetFwdRefVector fwdrr2_fft_4;
    reco::FFTPFJetFwdPtrVector fwdpr2_fft_4;
    reco::FFTPFJetRefProd rrr2_fft_4;
    edm::Wrapper<reco::FFTPFJetCollection> w2_fft_4;
    edm::Wrapper<reco::FFTPFJetRefVector> wrv2_fft_4;
    edm::Wrapper<reco::FFTPFJetFwdRefVector> wfwdrv2_fft_4;
    edm::Wrapper<reco::FFTPFJetFwdPtrVector> wfwdpv2_fft_4;
    edm::reftobase::Holder<reco::Candidate, reco::FFTPFJetRef> rtb2_fft_4;
    reco::JetTrackMatch<reco::FFTPFJetCollection> jtm2_fft_4;
    edm::reftobase::Holder<reco::Jet, reco::FFTPFJetRef> hgj_fft_4;
    edm::reftobase::RefHolder<reco::FFTPFJetRef> rhgj_fft_4;
    edm::Ptr<reco::FFTPFJet> ptrgj_fft_4;
    edm::PtrVector<reco::FFTPFJet> ptrvgj_fft_4;
    edm::Association<reco::FFTPFJetCollection> a_gj_fft_4;
    edm::Wrapper<edm::Association<reco::FFTPFJetCollection> > w_a_gj_fft_4;

    reco::FFTTrackJet jet_fft_6;
    reco::FFTTrackJetCollection o2_fft_6;
    reco::FFTTrackJetRef r2_fft_6;
    reco::FFTTrackJetFwdRef fwdr2_fft_6;
    reco::FFTTrackJetFwdPtr fwdp2_fft_6;
    reco::FFTTrackJetRefVector rr2_fft_6;
    reco::FFTTrackJetFwdRefVector fwdrr2_fft_6;
    reco::FFTTrackJetFwdPtrVector fwdpr2_fft_6;
    reco::FFTTrackJetRefProd rrr2_fft_6;
    edm::Wrapper<reco::FFTTrackJetCollection> w2_fft_6;
    edm::Wrapper<reco::FFTTrackJetRefVector> wrv2_fft_6;
    edm::Wrapper<reco::FFTTrackJetFwdRefVector> wfwdrv2_fft_6;
    edm::Wrapper<reco::FFTTrackJetFwdPtrVector> wfwdpv2_fft_6;
    edm::reftobase::Holder<reco::Candidate, reco::FFTTrackJetRef> rtb2_fft_6;
    reco::JetTrackMatch<reco::FFTTrackJetCollection> jtm2_fft_6;
    edm::reftobase::Holder<reco::Jet, reco::FFTTrackJetRef> hgj_fft_6;
    edm::reftobase::RefHolder<reco::FFTTrackJetRef> rhgj_fft_6;
    edm::Ptr<reco::FFTTrackJet> ptrgj_fft_6;
    edm::PtrVector<reco::FFTTrackJet> ptrvgj_fft_6;
    edm::Association<reco::FFTTrackJetCollection> a_gj_fft_6;
    edm::Wrapper<edm::Association<reco::FFTTrackJetCollection> > w_a_gj_fft_6;

    reco::FFTJPTJet jet_fft_7;
    reco::FFTJPTJetCollection o2_fft_7;
    reco::FFTJPTJetRef r2_fft_7;
    reco::FFTJPTJetFwdRef fwdr2_fft_7;
    reco::FFTJPTJetFwdPtr fwdp2_fft_7;
    reco::FFTJPTJetRefVector rr2_fft_7;
    reco::FFTJPTJetFwdRefVector fwdrr2_fft_7;
    reco::FFTJPTJetFwdPtrVector fwdpr2_fft_7;
    reco::FFTJPTJetRefProd rrr2_fft_7;
    edm::Wrapper<reco::FFTJPTJetCollection> w2_fft_7;
    edm::Wrapper<reco::FFTJPTJetRefVector> wrv2_fft_7;
    edm::Wrapper<reco::FFTJPTJetFwdRefVector> wfwdrv2_fft_7;
    edm::Wrapper<reco::FFTJPTJetFwdPtrVector> wfwdpv2_fft_7;
    edm::reftobase::Holder<reco::Candidate, reco::FFTJPTJetRef> rtb2_fft_7;
    reco::JetTrackMatch<reco::FFTJPTJetCollection> jtm2_fft_7;
    edm::reftobase::Holder<reco::Jet, reco::FFTJPTJetRef> hgj_fft_7;
    edm::reftobase::RefHolder<reco::FFTJPTJetRef> rhgj_fft_7;
    edm::Ptr<reco::FFTJPTJet> ptrgj_fft_7;
    edm::PtrVector<reco::FFTJPTJet> ptrvgj_fft_7;
    edm::Association<reco::FFTJPTJetCollection> a_gj_fft_7;
    edm::Wrapper<edm::Association<reco::FFTJPTJetCollection> > w_a_gj_fft_7;

    // Discretized energy flow
    reco::DiscretizedEnergyFlow r_dflow;
    edm::Wrapper<reco::DiscretizedEnergyFlow> wr_r_dflow;    

    // Pile-up summary
    reco::FFTJetPileupSummary r_fft_psumary;
    edm::Wrapper<reco::FFTJetPileupSummary> wr_r_fft_psumary;
  };
}
#endif
