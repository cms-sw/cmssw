#ifndef HitPairGeneratorFromLayerPairForPhotonConversion_h
#define HitPairGeneratorFromLayerPairForPhotonConversion_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "ConversionRegion.h"

class DetLayer;
class IdealMagneticFieldRecord;
class MagneticField;
class TrackingRegion;

class dso_hidden HitPairGeneratorFromLayerPairForPhotonConversion {  // : public HitPairGenerator {

public:
  typedef LayerHitMapCache LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;

  HitPairGeneratorFromLayerPairForPhotonConversion(edm::ConsumesCollector iC,
                                                   unsigned int inner,
                                                   unsigned int outer,
                                                   LayerCacheType* layerCache,
                                                   unsigned int nSize = 30000,
                                                   unsigned int max = 0);

  //  virtual ~HitPairGeneratorFromLayerPairForPhotonConversion() { }

  void hitPairs(const ConversionRegion& convRegion,
                const TrackingRegion& reg,
                OrderedHitPairs& prs,
                const Layers& layers,
                const edm::Event& ev,
                const edm::EventSetup& es);

  float getLayerRadius(const DetLayer& layer);
  float getLayerZ(const DetLayer& layer);

  bool checkBoundaries(const DetLayer& layer, const ConversionRegion& convRegion, float maxSearchR, float maxSearchZ);
  bool getPhiRange(float& Phimin,
                   float& Phimax,
                   const DetLayer& layer,
                   const ConversionRegion& convRegion,
                   const MagneticField& field);
  bool getPhiRange(float& Phimin,
                   float& Phimax,
                   const float& layerR,
                   const ConversionRegion& convRegion,
                   const MagneticField& field);
  bool checkRZCompatibilityWithSeedTrack(const RecHitsSortedInPhi::Hit& hit,
                                         const DetLayer& layer,
                                         const ConversionRegion& convRegion);

private:
  double getCot(double dz, double dr);

  LayerCacheType& theLayerCache;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;

  std::stringstream ss;
};

#endif
