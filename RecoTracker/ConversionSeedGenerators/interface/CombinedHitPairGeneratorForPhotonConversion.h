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
  CombinedHitPairGeneratorForPhotonConversion(const edm::ParameterSet & cfg);
  virtual ~CombinedHitPairGeneratorForPhotonConversion();

  void  add(const ctfseeding::SeedingLayer & inner, 
	      const ctfseeding::SeedingLayer & outer);

  /// form base class
  virtual void hitPairs(const TrackingRegion&, OrderedHitPairs&, const edm::Event&, const edm::EventSetup&){};

  /// from base class
  virtual CombinedHitPairGeneratorForPhotonConversion * clone() const 
    { return new CombinedHitPairGeneratorForPhotonConversion(theConfig); } 

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
  void init(const ctfseeding::SeedingLayerSets & layerSets);
  void init(const edm::ParameterSet & cfg, const edm::EventSetup& es);
  void cleanup();


  mutable bool initialised;
  edm::ParameterSet theConfig;
  uint32_t maxHitPairsPerTrackAndGenerator;

  LayerCacheType   theLayerCache;

  edm::ESWatcher<TrackerDigiGeometryRecord> theESWatcher;

  typedef std::vector<HitPairGeneratorFromLayerPairForPhotonConversion *>   Container;
  Container        theGenerators;

  OrderedHitPairs thePairs;

};
#endif
