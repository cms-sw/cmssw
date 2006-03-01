#include "FWCore/EDProduct/interface/Wrapper.h"

#include "DataFormats/BTauReco/interface/TrackTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackTagInfoFwd.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TauJetTag.h"
#include "DataFormats/BTauReco/interface/BJetTagCombined.h"
#include "DataFormats/BTauReco/interface/BJetTagTrackCounting.h"
#include "DataFormats/BTauReco/interface/BJetTagProbability.h"

#include <vector>

namespace {
  namespace {

    std::vector<reco::TrackTagInfo> v1;
    edm::Wrapper<std::vector<reco::TrackTagInfo> > c1;
    edm::Ref<std::vector<reco::TrackTagInfo> > r1;
    edm::RefVector<std::vector<reco::TrackTagInfo> > rv1;


    std::vector<reco::BJetTagCombined> v3;
    edm::Wrapper<std::vector<reco::BJetTagCombined> > c3;
    edm::Ref<std::vector<reco::BJetTagCombined> > r13;
    edm::RefVector<std::vector<reco::BJetTagCombined> > rv13;
    
    std::vector<reco::BJetTagTrackCounting> v4;
    edm::Wrapper<std::vector<reco::BJetTagTrackCounting> > c4;
    edm::Ref<std::vector<reco::BJetTagTrackCounting> > r14;
    edm::RefVector<std::vector<reco::BJetTagTrackCounting> > rv14;

     std::vector<reco::BJetTagProbability> v5;
    edm::Wrapper<std::vector<reco::BJetTagProbability> > c5;
    edm::Ref<std::vector<reco::BJetTagProbability> > r15;
    edm::RefVector<std::vector<reco::BJetTagProbability> > rv15;

    std::vector<reco::TauJetTag> v6;
    edm::Wrapper<std::vector<reco::TauJetTag> > c6;
    edm::Ref<std::vector<reco::TauJetTag> > r16;
    edm::RefVector<std::vector<reco::TauJetTag> > rv16;

 


  }
}
