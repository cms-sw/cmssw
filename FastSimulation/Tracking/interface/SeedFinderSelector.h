#ifndef SEEDFINDERSELECTOR_H
#define SEEDFINDERSELECTOR_H

#include <vector>
#include <memory>
#include <string>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

class TrackingRegion;
class FastTrackerRecHit;
class MultiHitGeneratorFromPairAndLayers;
class HitTripletGeneratorFromPairAndLayers;
class CAHitTripletGenerator;
class CAHitQuadrupletGenerator;
class IdealMagneticFieldRecord;
class MagneticField;
class MultipleScatteringParametrisationMaker;
class TrackerMultipleScatteringRecord;

class SeedFinderSelector {
public:
  SeedFinderSelector(const edm::ParameterSet&, edm::ConsumesCollector&&);

  ~SeedFinderSelector();

  void initEvent(const edm::Event&, const edm::EventSetup&);

  void setTrackingRegion(const TrackingRegion* trackingRegion) { trackingRegion_ = trackingRegion; }

  bool pass(const std::vector<const FastTrackerRecHit*>& hits) const;
  //new for Phase1
  SeedingLayerSetsBuilder::SeedingLayerId Layer_tuple(const FastTrackerRecHit* hit) const;

private:
  std::unique_ptr<HitTripletGeneratorFromPairAndLayers> pixelTripletGenerator_;
  std::unique_ptr<MultiHitGeneratorFromPairAndLayers> multiHitGenerator_;
  const TrackingRegion* trackingRegion_;
  const edm::EventSetup* eventSetup_;
  const MeasurementTracker* measurementTracker_;
  const std::string measurementTrackerLabel_;
  const edm::ESGetToken<MeasurementTracker, CkfComponentsRecord> measurementTrackerESToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyESToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldESToken_;
  const edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> msMakerESToken_;
  const TrackerTopology* trackerTopology_ = nullptr;
  const MagneticField* field_ = nullptr;
  const MultipleScatteringParametrisationMaker* msmaker_ = nullptr;
  std::unique_ptr<CAHitTripletGenerator> CAHitTriplGenerator_;
  std::unique_ptr<CAHitQuadrupletGenerator> CAHitQuadGenerator_;
  std::unique_ptr<SeedingLayerSetsBuilder> seedingLayers_;
  std::unique_ptr<SeedingLayerSetsHits> seedingLayer;
  std::vector<unsigned> layerPairs_;
  std::vector<SeedingLayerSetsBuilder::SeedingLayerId> seedingLayerIds;
};

#endif
