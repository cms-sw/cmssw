// -*- C++ -*-
//
// Package:    SiPixelTrackProbQXYProducer
// Class:      SiPixelTrackProbQXYProducer
//
/**\class SiPixelTrackProbQXYProducer  SiPixelTrackProbQXYProducer.cc RecoTracker/DeDx/plugins/SiPixelTrackProbQXYProducer.cc

 Description: SiPixel Charge and shape probabilities combined for tracks

*/
//
// Original Author:  Tamas Almos Vami
//         Created:  Mon Nov 17 14:09:02 CEST 2021
//

// system include files
#include <memory>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/SiPixelTrackProbQXY.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class declaration
//

class SiPixelTrackProbQXYProducer : public edm::global::EDProducer<> {
public:
  explicit SiPixelTrackProbQXYProducer(const edm::ParameterSet&);
  ~SiPixelTrackProbQXYProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDPutTokenT<edm::ValueMap<reco::SiPixelTrackProbQXY>> putProbQXYVMToken_;
  const edm::EDPutTokenT<edm::ValueMap<reco::SiPixelTrackProbQXY>> putProbQXYNoLayer1VMToken_;
};

using namespace reco;
using namespace std;
using namespace edm;

SiPixelTrackProbQXYProducer::SiPixelTrackProbQXYProducer(const edm::ParameterSet& iConfig)
    : tTopoToken_(esConsumes()),
      trackToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      putProbQXYVMToken_(produces<edm::ValueMap<reco::SiPixelTrackProbQXY>>()),
      putProbQXYNoLayer1VMToken_(produces<edm::ValueMap<reco::SiPixelTrackProbQXY>>("NoLayer1")) {}

void SiPixelTrackProbQXYProducer::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Retrieve track collection from the event
  auto trackCollectionHandle = iEvent.getHandle(trackToken_);
  const TrackCollection& trackCollection(*trackCollectionHandle.product());

  // Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(tTopoToken_);

  // Creates the output collection
  auto resultSiPixelTrackProbQXYColl = std::make_unique<reco::SiPixelTrackProbQXYCollection>();
  auto resultSiPixelTrackProbQXYNoLayer1Coll = std::make_unique<reco::SiPixelTrackProbQXYCollection>();

  // Loop through the tracks
  for (const auto& track : trackCollection) {
    float probQonTrack = 0.0;
    float probXYonTrack = 0.0;
    float probQonTrackNoLayer1 = 0.0;
    float probXYonTrackNoLayer1 = 0.0;
    int numRecHits = 0;
    int numRecHitsNoLayer1 = 0;
    float probQonTrackWMulti = 1;
    float probXYonTrackWMulti = 1;
    float probQonTrackWMultiNoLayer1 = 1;
    float probXYonTrackWMultiNoLayer1 = 1;
    // Loop through the rechits on the given track
    for (auto const& hit : track.recHits()) {
      const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit);
      if (pixhit == nullptr)
        continue;
      if (!pixhit->isValid())
        continue;

      // Have a separate variable that excludes Layer 1
      // Layer 1 was very noisy in 2017/2018
      if (pixhit->geographicalId().subdetId() == PixelSubdetector::PixelBarrel &&
          tTopo->pxbLayer(pixhit->geographicalId()) != 1) {
        float probQNoLayer1 = pixhit->probabilityQ();
        float probXYNoLayer1 = pixhit->probabilityXY();
        if (probQNoLayer1 != 0) {  // only save the non-zero rechits
          numRecHitsNoLayer1++;
          // Calculate alpha term needed for the combination
          probQonTrackWMultiNoLayer1 *= probQNoLayer1;
          probXYonTrackWMultiNoLayer1 *= probXYNoLayer1;
        }
      }

      // Have a variable that includes all layers and disks
      float probQ = pixhit->probabilityQ();
      float probXY = pixhit->probabilityXY();

      if (probQ == 0) {
        continue;  // if any of the rechits have zero probQ, skip them
      }
      numRecHits++;

      // Calculate alpha term needed for the combination
      probQonTrackWMulti *= probQ;
      probXYonTrackWMulti *= probXY;

    }  // end looping on the rechits

    // Combine the probabilities into a track level quantity
    float logprobQonTrackWMulti = probQonTrackWMulti > 0 ? log(probQonTrackWMulti) : 0;
    float logprobXYonTrackWMulti = probXYonTrackWMulti > 0 ? log(probXYonTrackWMulti) : 0;
    float factQ = -logprobQonTrackWMulti;
    float factXY = -logprobXYonTrackWMulti;
    float probQonTrackTerm = 1.f + factQ;
    float probXYonTrackTerm = 1.f + factXY;

    for (int iTkRh = 2; iTkRh < numRecHits; ++iTkRh) {
      factQ *= -logprobQonTrackWMulti / float(iTkRh);
      factXY *= -logprobXYonTrackWMulti / float(iTkRh);
      probQonTrackTerm += factQ;
      probXYonTrackTerm += factXY;
    }

    probQonTrack = probQonTrackWMulti * probQonTrackTerm;
    probXYonTrack = probXYonTrackWMulti * probXYonTrackTerm;

    // Repeat the above excluding Layer 1
    float logprobQonTrackWMultiNoLayer1 = probQonTrackWMultiNoLayer1 > 0 ? log(probQonTrackWMultiNoLayer1) : 0;
    float logprobXYonTrackWMultiNoLayer1 = probXYonTrackWMultiNoLayer1 > 0 ? log(probXYonTrackWMultiNoLayer1) : 0;

    float factQNoLayer1 = -logprobQonTrackWMultiNoLayer1;
    float factXYNoLayer1 = -logprobXYonTrackWMultiNoLayer1;
    float probQonTrackTermNoLayer1 = 1.f + factQNoLayer1;
    float probXYonTrackTermNoLayer1 = 1.f + factXYNoLayer1;

    for (int iTkRh = 2; iTkRh < numRecHits; ++iTkRh) {
      factQNoLayer1 *= -logprobQonTrackWMultiNoLayer1 / float(iTkRh);
      factXYNoLayer1 *= -logprobXYonTrackWMultiNoLayer1 / float(iTkRh);
      probQonTrackTermNoLayer1 += factQNoLayer1;
      probXYonTrackTermNoLayer1 += factXYNoLayer1;
    }

    probQonTrackNoLayer1 = probQonTrackWMultiNoLayer1 * probQonTrackTermNoLayer1;
    probXYonTrackNoLayer1 = probXYonTrackWMultiNoLayer1 * probXYonTrackTermNoLayer1;

    // Store the values in the collection
    resultSiPixelTrackProbQXYColl->emplace_back(probQonTrack, probXYonTrack);
    resultSiPixelTrackProbQXYNoLayer1Coll->emplace_back(probQonTrackNoLayer1, probXYonTrackNoLayer1);
  }  // end loop on track collection

  // Populate the event with the value map
  auto trackProbQXYMatch = std::make_unique<edm::ValueMap<reco::SiPixelTrackProbQXY>>();
  edm::ValueMap<reco::SiPixelTrackProbQXY>::Filler filler(*trackProbQXYMatch);
  filler.insert(trackCollectionHandle, resultSiPixelTrackProbQXYColl->begin(), resultSiPixelTrackProbQXYColl->end());
  filler.fill();
  iEvent.put(putProbQXYVMToken_, std::move(trackProbQXYMatch));

  auto trackProbQXYMatchNoLayer1 = std::make_unique<edm::ValueMap<reco::SiPixelTrackProbQXY>>();
  edm::ValueMap<reco::SiPixelTrackProbQXY>::Filler fillerNoLayer1(*trackProbQXYMatchNoLayer1);
  fillerNoLayer1.insert(trackCollectionHandle,
                        resultSiPixelTrackProbQXYNoLayer1Coll->begin(),
                        resultSiPixelTrackProbQXYNoLayer1Coll->end());
  fillerNoLayer1.fill();
  iEvent.put(putProbQXYNoLayer1VMToken_, std::move(trackProbQXYMatchNoLayer1));
}

void SiPixelTrackProbQXYProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Producer that creates SiPixel Charge and shape probabilities combined for tracks");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"))
      ->setComment("Input track collection for the producer");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelTrackProbQXYProducer);
