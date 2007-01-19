#include <utility>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
// #include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedSVTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedSVTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackTauImpactParameterAssociation.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"

namespace {
  namespace {
    reco::JetTagCollection v1;
    edm::Wrapper<reco::JetTagCollection> w1;
    edm::Ref<reco::JetTagCollection> r1;
    edm::RefProd<reco::JetTagCollection> rp1;
    edm::RefVector<reco::JetTagCollection> rv1;

    reco::TaggingVariableListCollection tvlc;
    edm::Wrapper<reco::TaggingVariableListCollection> tvlc_w;
    edm::Ref<reco::TaggingVariableListCollection> tvlc_r;
    edm::RefProd<reco::TaggingVariableListCollection> tvlc_rp;
    edm::RefVector<reco::TaggingVariableListCollection> tvlc_rv;

    reco::TrackCountingTagInfoCollection v2;
    edm::Wrapper<reco::TrackCountingTagInfoCollection> w2;
    edm::Ref<reco::TrackCountingTagInfoCollection> r2;
    edm::RefProd<reco::TrackCountingTagInfoCollection> rp2;
    edm::RefVector<reco::TrackCountingTagInfoCollection> rv2;

    reco::TrackProbabilityTagInfoCollection v2p;
    edm::Wrapper<reco::TrackProbabilityTagInfoCollection> w2p;
    edm::Ref<reco::TrackProbabilityTagInfoCollection> r2p;
    edm::RefProd<reco::TrackProbabilityTagInfoCollection> rp2p;
    edm::RefVector<reco::TrackProbabilityTagInfoCollection> rv2p;

    reco::JetTracksAssociationCollection v3;
    edm::Wrapper<reco::JetTracksAssociationCollection> w3;
    reco::JetTracksAssociation ra3;
    reco::JetTracksAssociationRef r3;
    reco::JetTracksAssociationRefProd rp3;
    reco::JetTracksAssociationRefVector rv3;

    reco::JetCrystalsAssociationCollection v11;
    edm::Wrapper<reco::JetCrystalsAssociationCollection> w11;
    reco::JetCrystalsAssociation ra11;
    reco::JetCrystalsAssociationRef r11;
    reco::JetCrystalsAssociationRefProd rp11;
    reco::JetCrystalsAssociationRefVector rv11;

    reco::TrackTauImpactParameterAssociationCollection c1;
    edm::Wrapper<reco::TrackTauImpactParameterAssociationCollection> wc1;

    reco::CombinedSVTagInfoCollection v4;
    reco::CombinedSVTagInfo iv4;
    edm::Wrapper<reco::CombinedSVTagInfoCollection> w4;
    edm::Ref<reco::CombinedSVTagInfoCollection> r4;
    edm::RefProd<reco::CombinedSVTagInfoCollection> rp4;
    edm::RefVector<reco::CombinedSVTagInfoCollection> rv4;

    reco::IsolatedTauTagInfoCollection v5;
    edm::Wrapper<reco::IsolatedTauTagInfoCollection> w5;
    edm::Ref<reco::IsolatedTauTagInfoCollection> r5;
    edm::RefProd<reco::IsolatedTauTagInfoCollection> rp5;
    edm::RefVector<reco::IsolatedTauTagInfoCollection> rv5;

    reco::EMIsolatedTauTagInfoCollection v10;
    edm::Wrapper<reco::EMIsolatedTauTagInfoCollection> w10;
    edm::Ref<reco::EMIsolatedTauTagInfoCollection> r10;
    edm::RefProd<reco::EMIsolatedTauTagInfoCollection> rp10;
    edm::RefVector<reco::EMIsolatedTauTagInfoCollection> rv10;

    reco::CombinedTauTagInfoCollection v12;
    edm::Wrapper<reco::CombinedTauTagInfoCollection> w12;
    edm::Ref<reco::CombinedTauTagInfoCollection> r12;
    edm::RefProd<reco::CombinedTauTagInfoCollection> rp12;
    edm::RefVector<reco::CombinedTauTagInfoCollection> rv12;

    reco::SoftLeptonProperties ext1;
    std::pair<reco::TrackRef, reco::SoftLeptonProperties> ep1;
    std::vector<std::pair<reco::TrackRef, reco::SoftLeptonProperties> > em1;
    //std::pair<unsigned int, reco::SoftLeptonProperties> ep1;
    //edm::AssociationMap<edm::OneToValue<std::vector<reco::Track>, reco::SoftLeptonProperties, unsigned int> > em1;

    reco::SoftLeptonTagInfoCollection v6;
    edm::Wrapper<reco::SoftLeptonTagInfoCollection> w6;
    edm::Ref<reco::SoftLeptonTagInfoCollection> r6;
    edm::RefProd<reco::SoftLeptonTagInfoCollection> rp6;
    edm::RefVector<reco::SoftLeptonTagInfoCollection> rv6;

    reco::TauImpactParameterInfoCollection v7;
    reco::TauImpactParameterTrackData ct7;
    edm::Wrapper<reco::TauImpactParameterInfoCollection> w7;
    edm::Ref<reco::TauImpactParameterInfoCollection> r7;
    edm::RefProd<reco::TauImpactParameterInfoCollection> rp7;
    edm::RefVector<reco::TauImpactParameterInfoCollection> rv7;

    reco::JetEisolAssociationCollection v8;
    edm::Wrapper<reco::JetEisolAssociationCollection> w8;
    reco::JetEisolAssociation ra8;
    reco::JetEisolAssociationRef r8;
    reco::JetEisolAssociationRefProd rp8;
    reco::JetEisolAssociationRefVector rv8;

    reco::TauMassTagInfoCollection v9;
    edm::Wrapper<reco::TauMassTagInfoCollection> w9;
    edm::Ref<reco::TauMassTagInfoCollection> r9;
    edm::RefProd<reco::TauMassTagInfoCollection> rp9;
    edm::RefVector<reco::TauMassTagInfoCollection> rv9;

  }
}
