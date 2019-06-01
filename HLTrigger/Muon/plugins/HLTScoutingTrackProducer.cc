// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingTrackProducer
//
/**\class HLTScoutingTrackProducer HLTScoutingTrackProducer.cc HLTScoutingTrackProducer.cc
Description: Producer for Scouting Tracks
*/
//

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingTrack.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class HLTScoutingTrackProducer : public edm::global::EDProducer<> {
public:
  explicit HLTScoutingTrackProducer(const edm::ParameterSet&);
  ~HLTScoutingTrackProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const final;

  const edm::EDGetTokenT<reco::TrackCollection> otherTrackCollection_;
};

//
// constructors and destructor
//
HLTScoutingTrackProducer::HLTScoutingTrackProducer(const edm::ParameterSet& iConfig)
    : otherTrackCollection_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("OtherTracks"))) {
  //register products
  produces<ScoutingTrackCollection>();
}

HLTScoutingTrackProducer::~HLTScoutingTrackProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingTrackProducer::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const {
  using namespace edm;

  std::unique_ptr<ScoutingTrackCollection> outTrack(new ScoutingTrackCollection());

  Handle<reco::TrackCollection> otherTrackCollection;

  if (iEvent.getByToken(otherTrackCollection_, otherTrackCollection)) {
    // Produce tracks in event
    for (auto& trk : *otherTrackCollection) {
      outTrack->emplace_back(trk.pt(),
                             trk.eta(),
                             trk.phi(),
                             trk.chi2(),
                             trk.ndof(),
                             trk.charge(),
                             trk.dxy(),
                             trk.dz(),
                             trk.hitPattern().numberOfValidPixelHits(),
                             trk.hitPattern().trackerLayersWithMeasurement(),
                             trk.hitPattern().numberOfValidStripHits(),
                             trk.qoverp(),
                             trk.lambda(),
                             trk.dxyError(),
                             trk.dzError(),
                             trk.qoverpError(),
                             trk.lambdaError(),
                             trk.phiError(),
                             trk.dsz(),
                             trk.dszError());
    }
  }

  iEvent.put(std::move(outTrack));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("OtherTracks", edm::InputTag("hltPixelTracksL3MuonNoVtx"));
  descriptions.add("hltScoutingTrackProducer", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTScoutingTrackProducer);
