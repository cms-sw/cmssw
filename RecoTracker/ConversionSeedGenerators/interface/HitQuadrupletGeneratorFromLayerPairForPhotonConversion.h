#ifndef HitQuadrupletGeneratorFromLayerPairForPhotonConversion_h
#define HitQuadrupletGeneratorFromLayerPairForPhotonConversion_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/ConversionSeedGenerators/interface/ConversionRegion.h"

class DetLayer;
class TrackingRegion;

class HitQuadrupletGeneratorFromLayerPairForPhotonConversion : public HitPairGenerator {

public:

  typedef CombinedHitPairGenerator::LayerCacheType       LayerCacheType;
  typedef ctfseeding::SeedingLayer Layer;
 
  HitQuadrupletGeneratorFromLayerPairForPhotonConversion(const Layer& inner,
				const Layer& outer,
				LayerCacheType* layerCache,
				unsigned int nSize=30000,
				unsigned int max=0);

  virtual ~HitQuadrupletGeneratorFromLayerPairForPhotonConversion() { }

  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs, 
			 const edm::Event & ev,  const edm::EventSetup& es);

  virtual HitQuadrupletGeneratorFromLayerPairForPhotonConversion* clone() const {
    return new HitQuadrupletGeneratorFromLayerPairForPhotonConversion(*this);
  }

  const Layer & innerLayer() const { return theInnerLayer; }
  const Layer & outerLayer() const { return theOuterLayer; }

  bool failCheckRZCompatibility(const RecHitsSortedInPhi::Hit & hit, const DetLayer& layer, const HitRZCompatibility *checkRZ, const TrackingRegion & region);
  //void checkPhiRange(double phi1, double phi2);

  bool failCheckSlopeTest(const RecHitsSortedInPhi::Hit & ohit, const RecHitsSortedInPhi::Hit & nohit, const RecHitsSortedInPhi::Hit & ihit, const RecHitsSortedInPhi::Hit & nihit, const TrackingRegion & region);
  void bubbleSortVsR(int n, double* ax, double* ay, double* aey);
  bool failCheckSegmentZCompatibility(double &rInn, double &zInnMin, double &zInnMax,
				      double &rInt, double &zIntMin, double &zIntMax,
				      double &rOut, double &zOutMin, double &zOutMax);
  double getZAtR(double &rInn, double &zInn, double &r, double &rOut, double &zOut);
  double verySimpleFit(int size, double* ax, double* ay, double* e2y, double& p0, double& e2p0, double& p1);
  double getSqrEffectiveErrorOnZ(const RecHitsSortedInPhi::Hit & hit, const TrackingRegion & region);
  double getEffectiveErrorOnZ(const RecHitsSortedInPhi::Hit & hit, const TrackingRegion & region);

private:
  
  LayerCacheType & theLayerCache;
  Layer theOuterLayer;  
  Layer theInnerLayer; 

  std::stringstream *ss;

};

#endif
