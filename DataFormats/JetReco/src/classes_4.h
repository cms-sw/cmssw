#ifndef JetObjects4_classes_h
#define JetObjects4_classes_h

#include "DataFormats/JetReco/interface/BasicJet.h" 
#include "Rtypes.h" 

#include "DataFormats/JetReco/interface/CaloJetCollection.h" 
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/PFClusterJetCollection.h"
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
#include "DataFormats/JetReco/interface/FFTGenJetCollection.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/FwdRef.h" 
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace DataFormats_JetReco {
  struct dictionary4 {
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

  };
}
#endif
