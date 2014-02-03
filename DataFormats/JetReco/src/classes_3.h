#ifndef JetObjects3_classes_h
#define JetObjects3_classes_h

#include "Rtypes.h" 

#include "DataFormats/JetReco/interface/BasicJetCollection.h" 
#include "DataFormats/JetReco/interface/CaloJetCollection.h" 
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/PFClusterJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

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
  struct dictionary3 {
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
    reco::JetTracksAssociation::value_type      jt_vt;

    edm::Wrapper<reco::JetExtendedAssociation::Container>  jea_c_w;
    reco::JetExtendedAssociation::Container       jea_c;
    reco::JetExtendedAssociation::Ref             jea_r;
    reco::JetExtendedAssociation::RefProd         jea_rp;
    reco::JetExtendedAssociation::RefVector       jea_rv;
    std::pair<edm::RefToBase<reco::Jet>,reco::JetExtendedAssociation::JetExtendedData> jea_pair;

    // RefToBase Holders for Jets
    edm::RefToBase<reco::Jet>  rtbj;
    edm::reftobase::IndirectHolder<reco::Jet> ihj;
    edm::reftobase::IndirectVectorHolder<reco::Jet> ihvj;
    edm::reftobase::Holder<reco::Jet, reco::CaloJetRef> hcj;
    edm::reftobase::Holder<reco::Jet, reco::JPTJetRef> hjptj;
    edm::reftobase::Holder<reco::Jet, reco::GenJetRef> hgj;
    edm::reftobase::Holder<reco::Jet, reco::PFJetRef> hpfj;
    edm::reftobase::Holder<reco::Jet, reco::TrackJetRef> htj;
    edm::reftobase::Holder<reco::Jet, reco::PFClusterJetRef> hpfcj;
    edm::reftobase::RefHolder<reco::CaloJetRef> rhcj;
    edm::reftobase::RefHolder<reco::JPTJetRef> rhjptj;
    edm::reftobase::RefHolder<reco::GenJetRef> rhgj;
    edm::reftobase::RefHolder<reco::PFJetRef> rhpfj;
    edm::reftobase::RefHolder<reco::TrackJetRef> rhtj;
    edm::reftobase::RefHolder<reco::PFClusterJetRef> rhpfcj;
    edm::RefToBaseVector<reco::Jet> jrtbv;
    edm::Wrapper<edm::RefToBaseVector<reco::Jet> > jrtbv_w;
    edm::reftobase::BaseVectorHolder<reco::Jet> * bvhj_p;    // pointer since it's pure virtual
  };
}
#endif
