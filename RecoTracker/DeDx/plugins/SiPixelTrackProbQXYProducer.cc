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
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/SiPixelTrackProbQXY.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//
// class declaration
//

class SiPixelTrackProbQXYProducer : public edm::global::EDProducer<> {
public:
  explicit SiPixelTrackProbQXYProducer(const edm::ParameterSet&);
  ~SiPixelTrackProbQXYProducer() override = default;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  int factorial(int) const;

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
};

using namespace reco;
using namespace std;
using namespace edm;

SiPixelTrackProbQXYProducer::SiPixelTrackProbQXYProducer(const edm::ParameterSet& iConfig)
    : tTopoToken_(esConsumes()),
      trackToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))) {
  produces<reco::SiPixelTrackProbQXYCollection>();
  produces<reco::SiPixelTrackProbQXYAssociation>();
}

void SiPixelTrackProbQXYProducer::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(trackToken_, trackCollectionHandle);
  const TrackCollection& trackCollection(*trackCollectionHandle.product());
  float probQonTrack = 0.0;
  float probXYonTrack = 0.0;
  float probQonTrackNoLayer1 = 0.0;
  float probXYonTrackNoLayer1 = 0.0;
  // creates the output collection
  auto resultSiPixelTrackProbQXYColl = std::make_unique<reco::SiPixelTrackProbQXYCollection>();
  std::vector<int> indices;

  // Loop through the tracks
  for (const auto& track : trackCollection) {
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
      if (pixhit->geographicalId().det() != DetId::Tracker)
        continue;

      //Retrieve tracker topology from geometry
      const TrackerTopology* const tTopo = &iSetup.getData(tTopoToken_);

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
    float probQonTrackTerm = 0;
    float probXYonTrackTerm = 0;
    for (int iTkRh = 0; iTkRh < numRecHits; ++iTkRh) {
      probQonTrackTerm += ((pow(-logprobQonTrackWMulti, iTkRh)) / (factorial(iTkRh)));
      probXYonTrackTerm += ((pow(-logprobXYonTrackWMulti, iTkRh)) / (factorial(iTkRh)));
    }

    probQonTrack = probQonTrackWMulti * probQonTrackTerm;
    probXYonTrack = probXYonTrackWMulti * probXYonTrackTerm;

    // Repeat the above excluding Layer 1
    float logprobQonTrackWMultiNoLayer1 = probQonTrackWMultiNoLayer1 > 0 ? log(probQonTrackWMultiNoLayer1) : 0;
    float logprobXYonTrackWMultiNoLayer1 = probXYonTrackWMultiNoLayer1 > 0 ? log(probXYonTrackWMultiNoLayer1) : 0;
    float probQonTrackTermNoLayer1 = 0;
    float probXYonTrackTermNoLayer1 = 0;
    for (int iTkRh = 0; iTkRh < numRecHitsNoLayer1; ++iTkRh) {
      probQonTrackTermNoLayer1 += ((pow(-logprobQonTrackWMultiNoLayer1, iTkRh)) / (factorial(iTkRh)));
      probXYonTrackTermNoLayer1 += ((pow(-logprobXYonTrackWMultiNoLayer1, iTkRh)) / (factorial(iTkRh)));
    }

    probQonTrackNoLayer1 = probQonTrackWMultiNoLayer1 * probQonTrackTermNoLayer1;
    probXYonTrackNoLayer1 = probXYonTrackWMultiNoLayer1 * probXYonTrackTermNoLayer1;

    reco::SiPixelTrackProbQXY siPixelTrackProbQXY =
        SiPixelTrackProbQXY(probQonTrack, probXYonTrack, probQonTrackNoLayer1, probXYonTrackNoLayer1);
    indices.push_back(resultSiPixelTrackProbQXYColl->size());
    resultSiPixelTrackProbQXYColl->push_back(siPixelTrackProbQXY);
  }  // end loop on track collection

  edm::OrphanHandle<reco::SiPixelTrackProbQXYCollection> siPixelTrackProbQXYCollHandle =
      iEvent.put(std::move(resultSiPixelTrackProbQXYColl));

  //create map passing the handle to the matched collection
  auto trackProbQXYMatch = std::make_unique<reco::SiPixelTrackProbQXYAssociation>(siPixelTrackProbQXYCollHandle);
  reco::SiPixelTrackProbQXYAssociation::Filler filler(*trackProbQXYMatch);
  filler.insert(trackCollectionHandle, indices.begin(), indices.end());
  filler.fill();
  iEvent.put(std::move(trackProbQXYMatch));
}

int SiPixelTrackProbQXYProducer::factorial(int n) const { return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n; }

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelTrackProbQXYProducer);
