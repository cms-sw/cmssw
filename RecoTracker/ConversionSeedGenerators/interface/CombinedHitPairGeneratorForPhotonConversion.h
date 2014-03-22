#ifndef CombinedHitPairGeneratorForPhotonConversion_H
#define CombinedHitPairGeneratorForPhotonConversion_H

#include <vector>
#include <memory>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"


class TrackingRegion;
class OrderedHitPairs;
class HitPairGeneratorFromLayerPairForPhotonConversion;
class SeedingLayerSetsHits;
namespace edm { class Event; class EventSetup; }

#include "RecoTracker/ConversionSeedGenerators/interface/ConversionRegion.h"

/** \class CombinedHitPairGeneratorForPhotonConversion
 * Hides set of HitPairGeneratorFromLayerPairForPhotonConversion generators.
 */

class CombinedHitPairGeneratorForPhotonConversion : public HitPairGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  virtual ~CombinedHitPairGeneratorForPhotonConversion();

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) override;

  /// form base class
  virtual void hitPairs(const TrackingRegion&, OrderedHitPairs&, const edm::Event&, const edm::EventSetup&){};

  /// from base class
  virtual CombinedHitPairGeneratorForPhotonConversion * clone() const 
    { return new CombinedHitPairGeneratorForPhotonConversion(*this); }

  /*Added to the original class*/
  const OrderedHitPairs & run(
			      const ConversionRegion& convRegion,
			      const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);

  void hitPairs( const ConversionRegion& convRegion, const TrackingRegion& reg, 
		 OrderedHitPairs & result, const edm::Event& ev, const edm::EventSetup& es);


  void clearLayerCache(){theLayerCache.clear();}

  /*------------------------*/
private:
  CombinedHitPairGeneratorForPhotonConversion(const CombinedHitPairGeneratorForPhotonConversion & cb); 

  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;
  uint32_t maxHitPairsPerTrackAndGenerator;

  LayerCacheType   theLayerCache;

  std::unique_ptr<HitPairGeneratorFromLayerPairForPhotonConversion> theGenerator;

  OrderedHitPairs thePairs;

};
#endif
