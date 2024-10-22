#include "FastSimulation/Tracking/interface/SeedFinderSelector.h"

// framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

// track reco
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoTracker/PixelSeeding/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/PixelSeeding/interface/CAHitTripletGenerator.h"
#include "RecoTracker/PixelSeeding/interface/CAHitQuadrupletGenerator.h"
#include "RecoTracker/PixelSeeding/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
// data formats
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

SeedFinderSelector::SeedFinderSelector(const edm::ParameterSet &cfg, edm::ConsumesCollector &&consumesCollector)
    : trackingRegion_(nullptr),
      eventSetup_(nullptr),
      measurementTracker_(nullptr),
      measurementTrackerLabel_(cfg.getParameter<std::string>("measurementTracker")),
      measurementTrackerESToken_(consumesCollector.esConsumes(edm::ESInputTag("", measurementTrackerLabel_))),
      trackerTopologyESToken_(consumesCollector.esConsumes()),
      fieldESToken_(consumesCollector.esConsumes()),
      msMakerESToken_(consumesCollector.esConsumes()) {
  if (cfg.exists("pixelTripletGeneratorFactory")) {
    const edm::ParameterSet &tripletConfig = cfg.getParameter<edm::ParameterSet>("pixelTripletGeneratorFactory");
    pixelTripletGenerator_ = HitTripletGeneratorFromPairAndLayersFactory::get()->create(
        tripletConfig.getParameter<std::string>("ComponentName"), tripletConfig, consumesCollector);
  }

  if (cfg.exists("MultiHitGeneratorFactory")) {
    const edm::ParameterSet &tripletConfig = cfg.getParameter<edm::ParameterSet>("MultiHitGeneratorFactory");
    multiHitGenerator_ = MultiHitGeneratorFromPairAndLayersFactory::get()->create(
        tripletConfig.getParameter<std::string>("ComponentName"), tripletConfig, consumesCollector);
  }

  if (cfg.exists("CAHitTripletGeneratorFactory")) {
    const edm::ParameterSet &tripletConfig = cfg.getParameter<edm::ParameterSet>("CAHitTripletGeneratorFactory");
    CAHitTriplGenerator_ = std::make_unique<CAHitTripletGenerator>(tripletConfig, consumesCollector);
    seedingLayers_ = std::make_unique<SeedingLayerSetsBuilder>(
        cfg,
        consumesCollector,
        //calling the new FastSim specific constructor to make SeedingLayerSetsHits pointer for triplet iterations
        edm::InputTag("fastTrackerRecHits"));
    layerPairs_ = cfg.getParameter<std::vector<unsigned>>("layerPairs");  //allowed layer pairs for CA triplets
  }

  if (cfg.exists("CAHitQuadrupletGeneratorFactory")) {
    const edm::ParameterSet &quadrupletConfig = cfg.getParameter<edm::ParameterSet>("CAHitQuadrupletGeneratorFactory");
    CAHitQuadGenerator_ = std::make_unique<CAHitQuadrupletGenerator>(quadrupletConfig, consumesCollector);
    //calling the new FastSim specific constructor to make SeedingLayerSetsHits pointer for quadruplet iterations
    seedingLayers_ =
        std::make_unique<SeedingLayerSetsBuilder>(cfg, consumesCollector, edm::InputTag("fastTrackerRecHits"));
    layerPairs_ = cfg.getParameter<std::vector<unsigned>>("layerPairs");  //allowed layer pairs for CA quadruplets
  }

  if ((pixelTripletGenerator_ && multiHitGenerator_) || (CAHitTriplGenerator_ && pixelTripletGenerator_) ||
      (CAHitTriplGenerator_ && multiHitGenerator_)) {
    throw cms::Exception("FastSimTracking")
        << "It is forbidden to specify together 'pixelTripletGeneratorFactory', 'CAHitTripletGeneratorFactory' and "
           "'MultiHitGeneratorFactory' in configuration of SeedFinderSelection";
  }
  if ((pixelTripletGenerator_ && CAHitQuadGenerator_) || (CAHitTriplGenerator_ && CAHitQuadGenerator_) ||
      (multiHitGenerator_ && CAHitQuadGenerator_)) {
    throw cms::Exception("FastSimTracking")
        << "It is forbidden to specify 'CAHitQuadrupletGeneratorFactory' together with 'pixelTripletGeneratorFactory', "
           "'CAHitTripletGeneratorFactory' or 'MultiHitGeneratorFactory' in configuration of SeedFinderSelection";
  }
}

SeedFinderSelector::~SeedFinderSelector() { ; }

void SeedFinderSelector::initEvent(const edm::Event &ev, const edm::EventSetup &es) {
  eventSetup_ = &es;

  measurementTracker_ = &es.getData(measurementTrackerESToken_);
  trackerTopology_ = &es.getData(trackerTopologyESToken_);
  field_ = &es.getData(fieldESToken_);
  msmaker_ = &es.getData(msMakerESToken_);

  if (multiHitGenerator_) {
    multiHitGenerator_->initES(es);
  }

  //for CA triplet iterations
  if (CAHitTriplGenerator_) {
    seedingLayer = seedingLayers_->makeSeedingLayerSetsHitsforFastSim(ev, es);
    seedingLayerIds = seedingLayers_->layers();
    CAHitTriplGenerator_->initEvent(ev, es);
  }
  //for CA quadruplet iterations
  if (CAHitQuadGenerator_) {
    seedingLayer = seedingLayers_->makeSeedingLayerSetsHitsforFastSim(ev, es);
    seedingLayerIds = seedingLayers_->layers();
    CAHitQuadGenerator_->initEvent(ev, es);
  }
}

bool SeedFinderSelector::pass(const std::vector<const FastTrackerRecHit *> &hits) const {
  if (!measurementTracker_ || !eventSetup_) {
    throw cms::Exception("FastSimTracking") << "ERROR: event not initialized";
  }
  if (!trackingRegion_) {
    throw cms::Exception("FastSimTracking") << "ERROR: trackingRegion not set";
  }

  // check the inner 2 hits
  if (hits.size() < 2) {
    throw cms::Exception("FastSimTracking") << "SeedFinderSelector::pass requires at least 2 hits";
  }
  const DetLayer *firstLayer =
      measurementTracker_->geometricSearchTracker()->detLayer(hits[0]->det()->geographicalId());
  const DetLayer *secondLayer =
      measurementTracker_->geometricSearchTracker()->detLayer(hits[1]->det()->geographicalId());

  std::vector<BaseTrackerRecHit const *> firstHits{hits[0]};
  std::vector<BaseTrackerRecHit const *> secondHits{hits[1]};

  const RecHitsSortedInPhi fhm(firstHits, trackingRegion_->origin(), firstLayer);
  const RecHitsSortedInPhi shm(secondHits, trackingRegion_->origin(), secondLayer);

  HitDoublets result(fhm, shm);
  HitPairGeneratorFromLayerPair::doublets(
      *trackingRegion_, *firstLayer, *secondLayer, fhm, shm, *field_, *msmaker_, 0, result);

  if (result.empty()) {
    return false;
  }

  // check the inner 3 hits
  if (pixelTripletGenerator_ || multiHitGenerator_ || CAHitTriplGenerator_) {
    if (hits.size() < 3) {
      throw cms::Exception("FastSimTracking")
          << "For the given configuration, SeedFinderSelector::pass requires at least 3 hits";
    }
    const DetLayer *thirdLayer =
        measurementTracker_->geometricSearchTracker()->detLayer(hits[2]->det()->geographicalId());
    std::vector<const DetLayer *> thirdLayerDetLayer(1, thirdLayer);
    std::vector<BaseTrackerRecHit const *> thirdHits{hits[2]};
    const RecHitsSortedInPhi thm(thirdHits, trackingRegion_->origin(), thirdLayer);
    const RecHitsSortedInPhi *thmp = &thm;

    if (pixelTripletGenerator_) {
      OrderedHitTriplets tripletresult;
      pixelTripletGenerator_->hitTriplets(
          *trackingRegion_, tripletresult, *eventSetup_, result, &thmp, thirdLayerDetLayer, 1);
      return !tripletresult.empty();
    } else if (multiHitGenerator_) {
      OrderedMultiHits tripletresult;
      multiHitGenerator_->hitTriplets(*trackingRegion_, tripletresult, result, &thmp, thirdLayerDetLayer, 1);
      return !tripletresult.empty();
    }
    //new for Phase1
    else if (CAHitTriplGenerator_) {
      if (!seedingLayer)
        throw cms::Exception("FastSimTracking") << "ERROR: SeedingLayers pointer not set for CATripletGenerator";

      SeedingLayerSetsHits &layers = *seedingLayer;
      //constructing IntermediateHitDoublets to be passed onto CAHitTripletGenerator::hitNtuplets()
      IntermediateHitDoublets ihd(&layers);
      const TrackingRegion &tr_ = *trackingRegion_;
      auto filler = ihd.beginRegion(&tr_);

      //forming the SeedingLayerId of the hits
      std::array<SeedingLayerSetsBuilder::SeedingLayerId, 3> hitPair;
      hitPair[0] = Layer_tuple(hits[0]);
      hitPair[1] = Layer_tuple(hits[1]);
      hitPair[2] = Layer_tuple(hits[2]);

      //extracting the DetLayer of the hits
      const DetLayer *fLayer =
          measurementTracker_->geometricSearchTracker()->detLayer(hits[0]->det()->geographicalId());
      const DetLayer *sLayer =
          measurementTracker_->geometricSearchTracker()->detLayer(hits[1]->det()->geographicalId());
      const DetLayer *tLayer =
          measurementTracker_->geometricSearchTracker()->detLayer(hits[2]->det()->geographicalId());

      //converting FastTrackerRecHit hits to BaseTrackerRecHit
      std::vector<BaseTrackerRecHit const *> fHits{hits[0]};
      std::vector<BaseTrackerRecHit const *> sHits{hits[1]};
      std::vector<BaseTrackerRecHit const *> tHits{hits[2]};

      //forming the SeedingLayerSet for the hit doublets
      SeedingLayerSetsHits::SeedingLayerSet pairCandidate1, pairCandidate2;
      for (SeedingLayerSetsHits::SeedingLayerSet ls : *seedingLayer) {
        SeedingLayerSetsHits::SeedingLayerSet pairCandidate;
        for (const auto p : layerPairs_) {
          pairCandidate = ls.slice(p, p + 2);
          if (p == 0 && hitPair[0] == seedingLayerIds[pairCandidate[0].index()] &&
              hitPair[1] == seedingLayerIds[pairCandidate[1].index()])
            pairCandidate1 = pairCandidate;
          if (p == 1 && hitPair[1] == seedingLayerIds[pairCandidate[0].index()] &&
              hitPair[2] == seedingLayerIds[pairCandidate[1].index()])
            pairCandidate2 = pairCandidate;
        }
      }

      //Important: hits of the layer to be added to LayerHitMapCache
      auto &layerCache = filler.layerHitMapCache();

      //doublets for CA triplets from the allowed layer pair combinations:(0,1),(1,2) and storing in filler
      const RecHitsSortedInPhi &firsthm = *layerCache.add(
          pairCandidate1[0], std::make_unique<RecHitsSortedInPhi>(fHits, trackingRegion_->origin(), fLayer));
      const RecHitsSortedInPhi &secondhm = *layerCache.add(
          pairCandidate1[1], std::make_unique<RecHitsSortedInPhi>(sHits, trackingRegion_->origin(), sLayer));
      HitDoublets res1(firsthm, secondhm);
      HitPairGeneratorFromLayerPair::doublets(
          *trackingRegion_, *fLayer, *sLayer, firsthm, secondhm, *field_, *msmaker_, 0, res1);
      filler.addDoublets(pairCandidate1, std::move(res1));
      const RecHitsSortedInPhi &thirdhm = *layerCache.add(
          pairCandidate2[1], std::make_unique<RecHitsSortedInPhi>(tHits, trackingRegion_->origin(), tLayer));
      HitDoublets res2(secondhm, thirdhm);
      HitPairGeneratorFromLayerPair::doublets(
          *trackingRegion_, *sLayer, *tLayer, secondhm, thirdhm, *field_, *msmaker_, 0, res2);
      filler.addDoublets(pairCandidate2, std::move(res2));

      std::vector<OrderedHitSeeds> tripletresult;
      tripletresult.resize(ihd.regionSize());
      for (auto &ntuplet : tripletresult)
        ntuplet.reserve(3);
      //calling the function from the class, modifies tripletresult
      CAHitTriplGenerator_->hitNtuplets(ihd, tripletresult, *seedingLayer);
      return !tripletresult[0].empty();
    }
  }
  //new for Phase1
  if (CAHitQuadGenerator_) {
    if (hits.size() < 4) {
      throw cms::Exception("FastSimTracking")
          << "For the given configuration, SeedFinderSelector::pass requires at least 4 hits";
    }

    if (!seedingLayer)
      throw cms::Exception("FastSimTracking") << "ERROR: SeedingLayers pointer not set for CAHitQuadrupletGenerator";

    SeedingLayerSetsHits &layers = *seedingLayer;
    //constructing IntermediateHitDoublets to be passed onto CAHitQuadrupletGenerator::hitNtuplets()
    IntermediateHitDoublets ihd(&layers);
    const TrackingRegion &tr_ = *trackingRegion_;
    auto filler = ihd.beginRegion(&tr_);

    //forming the SeedingLayerId of the hits
    std::array<SeedingLayerSetsBuilder::SeedingLayerId, 4> hitPair;
    hitPair[0] = Layer_tuple(hits[0]);
    hitPair[1] = Layer_tuple(hits[1]);
    hitPair[2] = Layer_tuple(hits[2]);
    hitPair[3] = Layer_tuple(hits[3]);

    //extracting the DetLayer of the hits
    const DetLayer *fLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[0]->det()->geographicalId());
    const DetLayer *sLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[1]->det()->geographicalId());
    const DetLayer *tLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[2]->det()->geographicalId());
    const DetLayer *frLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[3]->det()->geographicalId());

    //converting FastTrackerRecHit hits to BaseTrackerRecHit
    std::vector<BaseTrackerRecHit const *> fHits{hits[0]};
    std::vector<BaseTrackerRecHit const *> sHits{hits[1]};
    std::vector<BaseTrackerRecHit const *> tHits{hits[2]};
    std::vector<BaseTrackerRecHit const *> frHits{hits[3]};

    //forming the SeedingLayerSet for the hit doublets
    SeedingLayerSetsHits::SeedingLayerSet pairCandidate1, pairCandidate2, pairCandidate3;
    for (SeedingLayerSetsHits::SeedingLayerSet ls : *seedingLayer) {
      SeedingLayerSetsHits::SeedingLayerSet pairCandidate;
      for (const auto p : layerPairs_) {
        pairCandidate = ls.slice(p, p + 2);
        if (p == 0 && hitPair[0] == seedingLayerIds[pairCandidate[0].index()] &&
            hitPair[1] == seedingLayerIds[pairCandidate[1].index()])
          pairCandidate1 = pairCandidate;
        if (p == 1 && hitPair[1] == seedingLayerIds[pairCandidate[0].index()] &&
            hitPair[2] == seedingLayerIds[pairCandidate[1].index()])
          pairCandidate2 = pairCandidate;
        if (p == 2 && hitPair[2] == seedingLayerIds[pairCandidate[0].index()] &&
            hitPair[3] == seedingLayerIds[pairCandidate[1].index()])
          pairCandidate3 = pairCandidate;
      }
    }

    //Important: hits of the layer to be added to LayerHitMapCache
    auto &layerCache = filler.layerHitMapCache();

    //doublets for CA quadruplets from the allowed layer pair combinations:(0,1),(1,2),(2,3) and storing in filler
    const RecHitsSortedInPhi &firsthm = *layerCache.add(
        pairCandidate1[0], std::make_unique<RecHitsSortedInPhi>(fHits, trackingRegion_->origin(), fLayer));
    const RecHitsSortedInPhi &secondhm = *layerCache.add(
        pairCandidate1[1], std::make_unique<RecHitsSortedInPhi>(sHits, trackingRegion_->origin(), sLayer));
    HitDoublets res1(firsthm, secondhm);
    HitPairGeneratorFromLayerPair::doublets(
        *trackingRegion_, *fLayer, *sLayer, firsthm, secondhm, *field_, *msmaker_, 0, res1);
    filler.addDoublets(pairCandidate1, std::move(res1));
    const RecHitsSortedInPhi &thirdhm = *layerCache.add(
        pairCandidate2[1], std::make_unique<RecHitsSortedInPhi>(tHits, trackingRegion_->origin(), tLayer));
    HitDoublets res2(secondhm, thirdhm);
    HitPairGeneratorFromLayerPair::doublets(
        *trackingRegion_, *sLayer, *tLayer, secondhm, thirdhm, *field_, *msmaker_, 0, res2);
    filler.addDoublets(pairCandidate2, std::move(res2));
    const RecHitsSortedInPhi &fourthhm = *layerCache.add(
        pairCandidate3[1], std::make_unique<RecHitsSortedInPhi>(frHits, trackingRegion_->origin(), frLayer));
    HitDoublets res3(thirdhm, fourthhm);
    HitPairGeneratorFromLayerPair::doublets(
        *trackingRegion_, *tLayer, *frLayer, thirdhm, fourthhm, *field_, *msmaker_, 0, res3);
    filler.addDoublets(pairCandidate3, std::move(res3));

    std::vector<OrderedHitSeeds> quadrupletresult;
    quadrupletresult.resize(ihd.regionSize());
    for (auto &ntuplet : quadrupletresult)
      ntuplet.reserve(4);
    //calling the function from the class, modifies quadrupletresult
    CAHitQuadGenerator_->hitNtuplets(ihd, quadrupletresult, *seedingLayer);
    return !quadrupletresult[0].empty();
  }

  return true;
}

//new for Phase1
SeedingLayerSetsBuilder::SeedingLayerId SeedFinderSelector::Layer_tuple(const FastTrackerRecHit *hit) const {
  GeomDetEnumerators::SubDetector subdet = GeomDetEnumerators::invalidDet;
  TrackerDetSide side = TrackerDetSide::Barrel;
  int idLayer = 0;

  if ((hit->det()->geographicalId()).subdetId() == PixelSubdetector::PixelBarrel) {
    subdet = GeomDetEnumerators::PixelBarrel;
    side = TrackerDetSide::Barrel;
    idLayer = trackerTopology_->pxbLayer(hit->det()->geographicalId());
  } else if ((hit->det()->geographicalId()).subdetId() == PixelSubdetector::PixelEndcap) {
    subdet = GeomDetEnumerators::PixelEndcap;
    idLayer = trackerTopology_->pxfDisk(hit->det()->geographicalId());
    if (trackerTopology_->pxfSide(hit->det()->geographicalId()) == 1) {
      side = TrackerDetSide::NegEndcap;
    } else {
      side = TrackerDetSide::PosEndcap;
    }
  }
  return std::make_tuple(subdet, side, idLayer);
}
