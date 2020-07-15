#ifndef CombinedHitPairGeneratorForPhotonConversion_H
#define CombinedHitPairGeneratorForPhotonConversion_H

#include <vector>
#include <memory>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "HitPairGeneratorFromLayerPairForPhotonConversion.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Visibility.h"

class TrackingRegion;
class OrderedHitPairs;
class HitPairGeneratorFromLayerPairForPhotonConversion;
class SeedingLayerSetsHits;
namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

#include "ConversionRegion.h"

/** \class CombinedHitPairGeneratorForPhotonConversion
 * Hides set of HitPairGeneratorFromLayerPairForPhotonConversion generators.
 */

class dso_hidden CombinedHitPairGeneratorForPhotonConversion {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  const OrderedHitPairs& run(const ConversionRegion& convRegion,
                             const TrackingRegion& region,
                             const edm::Event& ev,
                             const edm::EventSetup& es);

private:
  void hitPairs(const ConversionRegion& convRegion,
                const TrackingRegion& reg,
                OrderedHitPairs& result,
                const edm::Event& ev,
                const edm::EventSetup& es);

public:
  void clearCache();

private:
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;

  uint32_t maxHitPairsPerTrackAndGenerator;

  LayerCacheType theLayerCache;

  std::unique_ptr<HitPairGeneratorFromLayerPairForPhotonConversion> theGenerator;

  OrderedHitPairs thePairs;

  unsigned int theMaxElement;
};
#endif
