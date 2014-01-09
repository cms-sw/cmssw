#ifndef CombinedHitQuadrupletGeneratorForPhotonConversion_H
#define CombinedHitQuadrupletGeneratorForPhotonConversion_H

#include <vector>
#include <memory>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackingRegion;
class OrderedHitPairs;
class HitQuadrupletGeneratorFromLayerPairForPhotonConversion;
namespace edm { class Event; class EventSetup; }

#include "RecoTracker/ConversionSeedGenerators/interface/ConversionRegion.h"

/** \class CombinedHitQuadrupletGeneratorForPhotonConversion
 * Hides set of HitQuadrupletGeneratorFromLayerPairForPhotonConversion generators.
 */

class CombinedHitQuadrupletGeneratorForPhotonConversion : public HitPairGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet & cfg);
  virtual ~CombinedHitQuadrupletGeneratorForPhotonConversion();

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) override;

  /// form base class
  virtual void hitPairs(const TrackingRegion&, OrderedHitPairs&, const edm::Event&, const edm::EventSetup&);

  /// from base class
  virtual CombinedHitQuadrupletGeneratorForPhotonConversion * clone() const 
    { return new CombinedHitQuadrupletGeneratorForPhotonConversion(*this); }

  const OrderedHitPairs & run(const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);


  void clearLayerCache(){theLayerCache.clear();}

  /*------------------------*/
private:
  CombinedHitQuadrupletGeneratorForPhotonConversion(const CombinedHitQuadrupletGeneratorForPhotonConversion & cb); 

  edm::InputTag theSeedingLayerSrc;

  LayerCacheType   theLayerCache;

  std::unique_ptr<HitQuadrupletGeneratorFromLayerPairForPhotonConversion> theGenerator;

  OrderedHitPairs thePairs;

};
#endif
