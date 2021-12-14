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
  float combineProbs(float probOnTrackWMulti, int numRecHits) const;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const bool debugFlag_;
  const double trackPtCut_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDPutTokenT<edm::ValueMap<reco::SiPixelTrackProbQXY>> putProbQXYVMToken_;
  const edm::EDPutTokenT<edm::ValueMap<reco::SiPixelTrackProbQXY>> putProbQXYNoLayer1VMToken_;
};

using namespace reco;
using namespace std;
using namespace edm;

SiPixelTrackProbQXYProducer::SiPixelTrackProbQXYProducer(const edm::ParameterSet& iConfig)
    : debugFlag_(iConfig.getUntrackedParameter<bool>("debug", false)),
      trackPtCut_(iConfig.getParameter<double>("trackPtCut")),
      tTopoToken_(esConsumes()),
      trackToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      putProbQXYVMToken_(produces<edm::ValueMap<reco::SiPixelTrackProbQXY>>()),
      putProbQXYNoLayer1VMToken_(produces<edm::ValueMap<reco::SiPixelTrackProbQXY>>("NoLayer1")) {}

void SiPixelTrackProbQXYProducer::produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Retrieve track collection from the event
  auto trackCollectionHandle = iEvent.getHandle(trackToken_);
  const TrackCollection& trackCollection(*trackCollectionHandle.product());
  int numTrack = 0;
  int numTrackWSmallProbQ = 0;

  // Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(tTopoToken_);

  // Creates the output collection
  auto resultSiPixelTrackProbQXYColl = std::make_unique<reco::SiPixelTrackProbQXYCollection>();
  auto resultSiPixelTrackProbQXYNoLayer1Coll = std::make_unique<reco::SiPixelTrackProbQXYCollection>();

  // Loop through the tracks
  if (debugFlag_) {
    LogPrint("SiPixelTrackProbQXYProducer") << "  -----------------------------------------------";
    LogPrint("SiPixelTrackProbQXYProducer") << "  For track " << numTrack;
  }
  for (const auto& track : trackCollection) {
    numTrack++;
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
      if ((pixhit->geographicalId().subdetId() == PixelSubdetector::PixelEndcap) ||
          (pixhit->geographicalId().subdetId() == PixelSubdetector::PixelBarrel &&
           tTopo->pxbLayer(pixhit->geographicalId()) != 1)) {
        float probQNoLayer1 = pixhit->probabilityQ();
        float probXYNoLayer1 = pixhit->probabilityXY();
        // only save the non-zero probQ rechits
        // but keep the zero probXY rechits
        if (probQNoLayer1 > 0.f) {
          numRecHitsNoLayer1++;
          // Calculate alpha term needed for the combination
          probQonTrackWMultiNoLayer1 *= probQNoLayer1;
          probXYonTrackWMultiNoLayer1 *= probXYNoLayer1;
          if (debugFlag_) {
            LogDebug("SiPixelTrackProbQXYProducer")
                << "    >>>> For rechit excluding Layer 1: " << numRecHitsNoLayer1 << " ProbQ = " << probQNoLayer1;
          }
        }
      }

      // Have a variable that includes all layers and disks
      float probQ = pixhit->probabilityQ();
      float probXY = pixhit->probabilityXY();

      if (probQ == 0) {
        continue;  // if any of the rechits have zero probQ, skip them
      }
      numRecHits++;

      if (debugFlag_) {
        LogDebug("SiPixelTrackProbQXYProducer") << "    >>>> For rechit: " << numRecHits << " ProbQ = " << probQ;
      }

      // Calculate alpha term needed for the combination
      probQonTrackWMulti *= probQ;
      probXYonTrackWMulti *= probXY;

    }  // end looping on the rechits

    // To save space let's zero out the low pt tracks
    // otherwise combine into track level quantity
    if (track.pt() < trackPtCut_) {
      probQonTrack = 0;
      probXYonTrack = 0;
      probQonTrackNoLayer1 = 0;
      probXYonTrackNoLayer1 = 0;
    } else {
      probQonTrack = combineProbs(probQonTrackWMulti, numRecHits);
      probXYonTrack = combineProbs(probXYonTrackWMulti, numRecHits);
      probQonTrackNoLayer1 = combineProbs(probQonTrackWMultiNoLayer1, numRecHitsNoLayer1);
      probXYonTrackNoLayer1 = combineProbs(probXYonTrackWMultiNoLayer1, numRecHitsNoLayer1);
    }

    // Count the number of tracks with small probQonTrack
    if (probQonTrack < 0.1) {
      numTrackWSmallProbQ++;
    }

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

float SiPixelTrackProbQXYProducer::combineProbs(float probOnTrackWMulti, int numRecHits) const {
  float logprobOnTrackWMulti = probOnTrackWMulti > 0 ? log(probOnTrackWMulti) : 0;
  float factQ = -logprobOnTrackWMulti;
  float probOnTrackTerm = 0.f;

  if (numRecHits == 1) {
    probOnTrackTerm = 1.f;
  } else if (numRecHits > 1) {
    probOnTrackTerm = 1.f + factQ;
    for (int iTkRh = 2; iTkRh < numRecHits; ++iTkRh) {
      factQ *= -logprobOnTrackWMulti / float(iTkRh);
      probOnTrackTerm += factQ;
    }
  }
  float probOnTrack = probOnTrackWMulti * probOnTrackTerm;

  if (debugFlag_) {
    LogPrint("SiPixelTrackProbQXYProducer")
        << "  >> probOnTrack = " << probOnTrack << " = " << probOnTrackWMulti << " * " << probOnTrackTerm;
  }
  return probOnTrack;
}

void SiPixelTrackProbQXYProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Producer that creates SiPixel Charge and shape probabilities combined for tracks");
  desc.addUntracked<bool>("debug", false);
  desc.add<double>("trackPtCut", 5.0)->setComment("Cut on the pt of the track above which we store the probs");
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"))
      ->setComment("Input track collection for the producer");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelTrackProbQXYProducer);
