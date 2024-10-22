// -*- C++ -*-
//
// Package:    Calibration/TkAlCaRecoProducers
// Class:      TrackDistanceValueMapProducer
//
/**\class TrackDistanceValueMapProducer TrackDistanceValueMapProducer.cc Calibration/TkAlCaRecoProducers/plugins/TrackDistanceValueMapProducer.cc

 Description: creates a value map for each saved muon track with all the distances of the other track w.r.t. the muon track 

*/
//
// Original Author:  Marco Musich
//         Created:  Mon, 12 Apr 2021 11:59:39 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Math/interface/deltaR.h"
//
// class declaration
//

class TrackDistanceValueMapProducer : public edm::global::EDProducer<> {
public:
  explicit TrackDistanceValueMapProducer(const edm::ParameterSet&);
  ~TrackDistanceValueMapProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  // edToken
  edm::EDGetTokenT<edm::View<reco::Track>> muonTracksToken_;
  edm::EDGetTokenT<edm::View<reco::Track>> otherTracksToken_;

  // save in the event only up to the n-th closest
  unsigned int nthClosestTrack_;

  // putToken
  edm::EDPutTokenT<edm::ValueMap<std::vector<float>>> distancesPutToken_;
};

//
// constructors and destructor
//
TrackDistanceValueMapProducer::TrackDistanceValueMapProducer(const edm::ParameterSet& iConfig)
    : muonTracksToken_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("muonTracks"))),
      otherTracksToken_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("allTracks"))),
      nthClosestTrack_(iConfig.getParameter<unsigned int>("saveUpToNthClosest")),
      distancesPutToken_(produces<edm::ValueMap<std::vector<float>>>()) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackDistanceValueMapProducer::produce(edm::StreamID streamID,
                                            edm::Event& iEvent,
                                            const edm::EventSetup& iSetup) const {
  using namespace edm;

  //=======================================================
  // Retrieve the muon Track information
  //=======================================================

  const auto& muonTrackCollectionHandle = iEvent.getHandle(muonTracksToken_);
  if (!muonTrackCollectionHandle.isValid())
    return;
  auto const& muonTracks = *muonTrackCollectionHandle;

  //=======================================================
  // Retrieve the general Track information
  //=======================================================

  const auto& allTrackCollectionHandle = iEvent.getHandle(otherTracksToken_);
  if (!allTrackCollectionHandle.isValid())
    return;
  auto const& allTracks = *allTrackCollectionHandle;

  //=======================================================
  // fill the distance vector
  //=======================================================

  // the map cannot be filled straight away, so create an intermediate vector
  unsigned int Nit = muonTracks.size();
  unsigned int Nall = allTracks.size();
  std::vector<std::vector<float>> v2_dR2;

  for (unsigned int iit = 0; iit < Nit; iit++) {
    const auto& muontrack = muonTracks.ptrAt(iit);

    std::vector<float> v_dR2;
    for (unsigned int iAll = 0; iAll < Nall; iAll++) {
      const auto& recotrack = allTracks.ptrAt(iAll);
      const float dR2 = ::deltaR2(*muontrack, *recotrack);
      if (dR2 != 0.f) {  // exclude the track itself
        v_dR2.push_back(dR2);
      }
    }

    // sort the tracks in ascending order of distance
    std::sort(v_dR2.begin(), v_dR2.end(), [](const float& lhs, const float& rhs) { return lhs < rhs; });

    // just copy the first nth
    std::vector<float> reduced_vdR2;
    std::copy(v_dR2.begin(),
              v_dR2.begin() + std::min(v_dR2.size(), static_cast<size_t>(nthClosestTrack_)),
              std::back_inserter(reduced_vdR2));
    v2_dR2.push_back(reduced_vdR2);
  }

  //=======================================================
  // Populate the event with the value map
  //=======================================================

  std::unique_ptr<edm::ValueMap<std::vector<float>>> vm_dR2(new edm::ValueMap<std::vector<float>>());
  edm::ValueMap<std::vector<float>>::Filler filler(*vm_dR2);
  filler.insert(muonTrackCollectionHandle, v2_dR2.begin(), v2_dR2.end());
  filler.fill();
  iEvent.put(distancesPutToken_, std::move(vm_dR2));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackDistanceValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Produces a value map with all the distances with the other tracks in the event");
  desc.add<edm::InputTag>("muonTracks", edm::InputTag("ALCARECOSiPixelCalSingleMuonTight"))
      ->setComment("the probe muon tracks");
  desc.add<edm::InputTag>("allTracks", edm::InputTag("generalTracks"))->setComment("all tracks in the event");
  desc.add<unsigned int>("saveUpToNthClosest", 1)->setComment("save the distance only for the nth closest tracks");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackDistanceValueMapProducer);
