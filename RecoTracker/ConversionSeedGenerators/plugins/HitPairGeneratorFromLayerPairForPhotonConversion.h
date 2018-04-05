#ifndef HitPairGeneratorFromLayerPairForPhotonConversion_h
#define HitPairGeneratorFromLayerPairForPhotonConversion_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "ConversionRegion.h"

class DetLayer;
class TrackingRegion;

class dso_hidden HitPairGeneratorFromLayerPairForPhotonConversion { // : public HitPairGenerator {

public:

  typedef LayerHitMapCache LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;
 
  HitPairGeneratorFromLayerPairForPhotonConversion(unsigned int inner,
                                unsigned int outer,
				LayerCacheType* layerCache,
				unsigned int nSize=30000,
				unsigned int max=0);

//  virtual ~HitPairGeneratorFromLayerPairForPhotonConversion() { }

  void hitPairs( const ConversionRegion& convRegion, const TrackingRegion& reg, OrderedHitPairs & prs, 
                 const Layers& layers,
			 const edm::Event & ev,  const edm::EventSetup& es);

  float getLayerRadius(const DetLayer& layer);
  float getLayerZ(const DetLayer& layer);

  bool checkBoundaries(const DetLayer& layer,const ConversionRegion& convRegion,float maxSearchR, float maxSearchZ);
  bool getPhiRange(float& Phimin, float& Phimax,const DetLayer& layer, const ConversionRegion &convRegion, const edm::EventSetup& es);
  bool getPhiRange(float& Phimin, float& Phimax, const float& layerR, const ConversionRegion &convRegion, const edm::EventSetup& es);
  bool checkRZCompatibilityWithSeedTrack(const RecHitsSortedInPhi::Hit & hit, const DetLayer& layer, const ConversionRegion& convRegion);


private:
  
  double getCot(double dz, double dr);
  
  LayerCacheType & theLayerCache;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;

  std::stringstream ss;

};

#endif
