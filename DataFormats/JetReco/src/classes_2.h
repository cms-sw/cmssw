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
#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace {
  struct dictionary {
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
    
    // jet -> PFCandidate associations, needed for boosted tau reconstruction
    edm::AssociationMap<edm::OneToMany<std::vector<reco::PFJet>,std::vector<reco::PFCandidate>,unsigned int> > jetPFCandidateAssociation_o;
    edm::Wrapper<edm::AssociationMap<edm::OneToMany<std::vector<reco::PFJet>,std::vector<reco::PFCandidate>,unsigned int> > > jetPFCandidateAssociation_w;
    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::PFJet> >,edm::RefProd<std::vector<reco::PFCandidate> > > jetPFCandidateAssociation_kv;
    edm::helpers::KeyVal<edm::Ref<std::vector<reco::PFJet>,reco::PFJet,edm::refhelper::FindUsingAdvance<std::vector<reco::PFJet>,reco::PFJet> >,edm::RefVector<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > jetPFCandidateAssociation_kv2;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::PFJet>,reco::PFJet,edm::refhelper::FindUsingAdvance<std::vector<reco::PFJet>,reco::PFJet> >,edm::RefVector<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > > jetPFCandidateAssociation_mkv;
  };
}
#endif
