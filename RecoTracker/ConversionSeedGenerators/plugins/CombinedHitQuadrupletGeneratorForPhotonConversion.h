#ifndef CombinedHitQuadrupletGeneratorForPhotonConversion_H
#define CombinedHitQuadrupletGeneratorForPhotonConversion_H

#include <vector>
#include <memory>
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class TrackingRegion;
class OrderedHitPairs;
class HitQuadrupletGeneratorFromLayerPairForPhotonConversion;
class SeedingLayerSetsHits;
namespace edm { class Event; class EventSetup; class ParameterSet; class ConsumesCollector;}

#include "ConversionRegion.h"

/** \class CombinedHitQuadrupletGeneratorForPhotonConversion
 * Hides set of HitQuadrupletGeneratorFromLayerPairForPhotonConversion generators.
 */

class CombinedHitQuadrupletGeneratorForPhotonConversion {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  ~CombinedHitQuadrupletGeneratorForPhotonConversion();

  void hitPairs(const TrackingRegion&, OrderedHitPairs&, const edm::Event&, const edm::EventSetup&);

  const OrderedHitPairs & run(const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);


  void clearLayerCache(){theLayerCache.clear();}

  /*------------------------*/
private:
  CombinedHitQuadrupletGeneratorForPhotonConversion(const CombinedHitQuadrupletGeneratorForPhotonConversion & cb); 

  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;
  const unsigned int theMaxElement;
  LayerCacheType   theLayerCache;

  std::unique_ptr<HitQuadrupletGeneratorFromLayerPairForPhotonConversion> theGenerator;

  OrderedHitPairs thePairs;

};
#endif
