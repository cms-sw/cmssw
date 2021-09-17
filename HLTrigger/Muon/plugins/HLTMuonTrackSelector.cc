/*
* class HLTMuonTrackSelector
* 
* See header file for documentation
*  
* Author: Kyeongpil Lee (kplee@cern.ch)
* 
*/

#include "HLTMuonTrackSelector.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace reco;

HLTMuonTrackSelector::HLTMuonTrackSelector(const edm::ParameterSet& iConfig)
    : collectionCloner(producesCollector(), iConfig, true),
      collectionClonerTokens(iConfig.getParameter<edm::InputTag>("track"), consumesCollector()),
      token_muon(consumes<vector<reco::Muon> >(iConfig.getParameter<edm::InputTag>("muon"))),
      token_originalMVAVals(consumes<MVACollection>(iConfig.getParameter<edm::InputTag>("originalMVAVals"))),
      flag_copyMVA(iConfig.getParameter<bool>("copyMVA")) {
  produces<MVACollection>("MVAValues");
}

HLTMuonTrackSelector::~HLTMuonTrackSelector() {}

void HLTMuonTrackSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("track", edm::InputTag());
  desc.add<edm::InputTag>("muon", edm::InputTag());
  desc.add<edm::InputTag>("originalMVAVals", edm::InputTag());
  desc.add<bool>("copyMVA", false);
  TrackCollectionCloner::fill(desc);  // -- add copyExtras and copyTrajectories
  descriptions.add("HLTMuonTrackSelector", desc);
}

void HLTMuonTrackSelector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  TrackCollectionCloner::Producer producer(iEvent, collectionCloner);

  // -- load tracks
  auto const& originalTracks = collectionClonerTokens.tracks(iEvent);
  auto nTrack = originalTracks.size();

  // -- load muons
  edm::Handle<vector<reco::Muon> > handle_muon;
  iEvent.getByToken(token_muon, handle_muon);
  auto nMuon = handle_muon->size();

  // -- load MVA values if necessary
  edm::Handle<MVACollection> handle_originalMVAVals;
  if (flag_copyMVA) {
    iEvent.getByToken(token_originalMVAVals, handle_originalMVAVals);
    assert((*handle_originalMVAVals).size() == nTrack);
  }

  // -- containers for selected track informations
  std::vector<unsigned int> selectedIter;
  auto selectedMVAVals = std::make_unique<MVACollection>();

  auto nSelected = 0U;

  ////////////////////
  // -- matching -- //
  ////////////////////

  // -- iteration over muons
  for (auto i_mu = 0U; i_mu < nMuon; ++i_mu) {
    // -- avoids crashing in case the muon is SA only.
    const reco::Muon& muon(handle_muon->at(i_mu));
    TrackRef muonTrackRef = (muon.innerTrack().isNonnull()) ? muon.innerTrack() : muon.muonBestTrack();

    double muonPt = muonTrackRef->pt();
    double muonEta = muonTrackRef->eta();
    double muonPhi = muonTrackRef->phi();

    double smallestDPt = 1e30;
    unsigned int smallestDPtIter = 9999U;

    // -- iteration over tracks
    for (auto i_trk = 0U; i_trk < nTrack; ++i_trk) {
      auto const& track = originalTracks[i_trk];

      double trackPt = track.pt();
      double trackEta = track.eta();
      double trackPhi = track.phi();

      if (deltaR(trackEta, trackPhi, muonEta, muonPhi) < 0.1) {
        double dPt = fabs(trackPt - muonPt);

        if (dPt < smallestDPt) {
          smallestDPt = dPt;
          smallestDPtIter = i_trk;
        }
      }
    }  // -- end of track iteration

    // -- if at least one track is matched
    if (smallestDPtIter != 9999U) {
      selectedIter.push_back(smallestDPtIter);
      if (flag_copyMVA)
        selectedMVAVals->push_back((*handle_originalMVAVals)[smallestDPtIter]);
      ++nSelected;
    }

  }  // -- end of muon iteration

  assert(producer.selTracks_->empty());

  // -- produces tracks and associated informations
  producer(collectionClonerTokens, selectedIter);
  assert(producer.selTracks_->size() == nSelected);

  if (flag_copyMVA)
    iEvent.put(std::move(selectedMVAVals), "MVAValues");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HLTMuonTrackSelector);
