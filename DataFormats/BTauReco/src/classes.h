#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/CombinedBTagInfo.h"

namespace {
  namespace {
    reco::JetTagCollection v1;   
    edm::Wrapper<reco::JetTagCollection> w1;
    edm::Ref<reco::JetTagCollection> r1;
    edm::RefProd<reco::JetTagCollection> rp1;
    edm::RefVector<reco::JetTagCollection> rv1;
    
    reco::TrackCountingTagInfoCollection v2;
    edm::Wrapper<reco::TrackCountingTagInfoCollection> w2;
    edm::Ref<reco::TrackCountingTagInfoCollection> r2;
    edm::RefProd<reco::TrackCountingTagInfoCollection> rp2;
    edm::RefVector<reco::TrackCountingTagInfoCollection> rv2;

    reco::JetTracksAssociationCollection v3;
    edm::Wrapper<reco::JetTracksAssociationCollection> w3;
    reco::JetTracksAssociation ra3;
    reco::JetTracksAssociationRef r3;
    reco::JetTracksAssociationRefProd rp3;
    reco::JetTracksAssociationRefVector rv3;

    reco::CombinedBTagInfoCollection v4;
    edm::Wrapper<reco::CombinedBTagInfoCollection> w4;
    edm::Ref<reco::CombinedBTagInfoCollection> r4;
    edm::RefProd<reco::CombinedBTagInfoCollection> rp4;
    edm::RefVector<reco::CombinedBTagInfoCollection> rv4;
  }
}
