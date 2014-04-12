#ifndef HitPairGeneratorFromLayerPairForPhotonConversion_h
#define HitPairGeneratorFromLayerPairForPhotonConversion_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/ConversionSeedGenerators/interface/ConversionRegion.h"

class DetLayer;
class TrackingRegion;

class HitPairGeneratorFromLayerPairForPhotonConversion : public HitPairGenerator {

public:

  typedef CombinedHitPairGenerator::LayerCacheType       LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;
 
  HitPairGeneratorFromLayerPairForPhotonConversion(unsigned int inner,
                                unsigned int outer,
				LayerCacheType* layerCache,
				unsigned int nSize=30000,
				unsigned int max=0);

  virtual ~HitPairGeneratorFromLayerPairForPhotonConversion() { }

  void setSeedingLayers(Layers layers) override { theSeedingLayers = layers; }

  void hitPairs( const ConversionRegion& convRegion, const TrackingRegion& reg, OrderedHitPairs & prs, 
			 const edm::Event & ev,  const edm::EventSetup& es);

  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs, 
			 const edm::Event & ev,  const edm::EventSetup& es){};

  virtual HitPairGeneratorFromLayerPairForPhotonConversion* clone() const {
    return new HitPairGeneratorFromLayerPairForPhotonConversion(*this);
  }

  Layer innerLayer() const { return theSeedingLayers[theInnerLayer]; }
  Layer outerLayer() const { return theSeedingLayers[theOuterLayer]; }

  float getLayerRadius(const DetLayer& layer);
  float getLayerZ(const DetLayer& layer);

  bool checkBoundaries(const DetLayer& layer,const ConversionRegion& convRegion,float maxSearchR, float maxSearchZ);
  bool getPhiRange(float& Phimin, float& Phimax,const DetLayer& layer, const ConversionRegion &convRegion, const edm::EventSetup& es);
  bool getPhiRange(float& Phimin, float& Phimax, const float& layerR, const ConversionRegion &convRegion, const edm::EventSetup& es);
  bool checkRZCompatibilityWithSeedTrack(const RecHitsSortedInPhi::Hit & hit, const DetLayer& layer, const ConversionRegion& convRegion);


private:
  
  double getCot(double dz, double dr);
  
  LayerCacheType & theLayerCache;
  Layers theSeedingLayers;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;

  std::stringstream *ss;

};

#endif
