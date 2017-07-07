#ifndef _PixelTripletLowPtGenerator_h_
#define _PixelTripletLowPtGenerator_h_

/** A HitTripletGenerator from HitPairGenerator and vector of
    Layers. The HitPairGenerator provides a set of hit pairs.
    For each pair the search for compatible hit(s) is done among
    provided Layers
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TripletFilter.h"

class TrackerGeometry;
class TripletFilter;
class SiPixelClusterShapeCache;

#include <vector>

class   PixelTripletLowPtGenerator :
 public HitTripletGeneratorFromPairAndLayers {


 public:
   PixelTripletLowPtGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelTripletLowPtGenerator() override;

  void hitTriplets( const TrackingRegion& region, OrderedHitTriplets & trs,
                            const edm::Event & ev, const edm::EventSetup& es,
                            const SeedingLayerSetsHits::SeedingLayerSet& pairLayers,
                            const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers) override;
  void hitTriplets(
		   const TrackingRegion& region, 
		   OrderedHitTriplets & result,
		   const edm::EventSetup & es,
		   const HitDoublets & doublets,
		   const RecHitsSortedInPhi ** thirdHitMap,
		   const std::vector<const DetLayer *> & thirdLayerDetLayer,
		   const int nThirdLayers)override;

 private:
  void getTracker (const edm::EventSetup& es);
  GlobalPoint getGlobalPosition(const TrackingRecHit* recHit);

  const TrackerGeometry * theTracker;
  std::unique_ptr<TripletFilter> theFilter;


  edm::EDGetTokenT<SiPixelClusterShapeCache> theClusterShapeCacheToken;
  double nSigMultipleScattering;
  double rzTolerance;
  double maxAngleRatio;

  std::string builderName;
  bool checkMultipleScattering;
  bool checkClusterShape;
 
};

#endif
