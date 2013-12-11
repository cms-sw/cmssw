#ifndef CombinedHitPairGeneratorForPhotonConversion_H
#define CombinedHitPairGeneratorForPhotonConversion_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackingRegion;
class OrderedHitPairs;
class HitPairGeneratorFromLayerPairForPhotonConversion;
namespace ctfseeding { class SeedingLayer;}
namespace edm { class Event; class EventSetup; }

#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

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
  CombinedHitPairGeneratorForPhotonConversion(const CombinedHitPairGeneratorForPhotonConversion & cb);

//   void  add(const ctfseeding::SeedingLayer & inner, 
// 	      const ctfseeding::SeedingLayer & outer);

  /// form base class
  virtual void hitPairs(const TrackingRegion&, OrderedHitPairs&, const edm::Event&, const edm::EventSetup&){};

  /// from base class
  CombinedHitPairGeneratorForPhotonConversion * clone() const override
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
  uint32_t maxHitPairsPerTrackAndGenerator;

  LayerCacheType   theLayerCache;

  typedef std::vector<std::unique_ptr<HitPairGeneratorFromLayerPairForPhotonConversion> >   Container;
  Container        theGenerators;

  OrderedHitPairs thePairs;

};
#endif
