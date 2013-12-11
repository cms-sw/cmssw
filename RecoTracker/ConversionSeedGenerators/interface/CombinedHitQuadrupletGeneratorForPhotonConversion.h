#ifndef CombinedHitQuadrupletGeneratorForPhotonConversion_H
#define CombinedHitQuadrupletGeneratorForPhotonConversion_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackingRegion;
class OrderedHitPairs;
class HitQuadrupletGeneratorFromLayerPairForPhotonConversion;
namespace ctfseeding { class SeedingLayer;}
namespace edm { class Event; class EventSetup; }

#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSets.h"

#include "RecoTracker/ConversionSeedGenerators/interface/ConversionRegion.h"

/** \class CombinedHitQuadrupletGeneratorForPhotonConversion
 * Hides set of HitQuadrupletGeneratorFromLayerPairForPhotonConversion generators.
 */

class CombinedHitQuadrupletGeneratorForPhotonConversion : public HitPairGenerator {
public:
  typedef LayerHitMapCache LayerCacheType;

public:
  CombinedHitQuadrupletGeneratorForPhotonConversion(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC);
  CombinedHitQuadrupletGeneratorForPhotonConversion(const CombinedHitQuadrupletGeneratorForPhotonConversion & cb);
  virtual ~CombinedHitQuadrupletGeneratorForPhotonConversion();

  /// form base class
  virtual void hitPairs(const TrackingRegion&, OrderedHitPairs&, const edm::Event&, const edm::EventSetup&);

  /// from base class
  virtual CombinedHitQuadrupletGeneratorForPhotonConversion * clone() const 
    { return new CombinedHitQuadrupletGeneratorForPhotonConversion(*this); }

  const OrderedHitPairs & run(const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es);


  void clearLayerCache(){theLayerCache.clear();}

  /*------------------------*/
private:
  LayerCacheType   theLayerCache;

  typedef std::vector<HitQuadrupletGeneratorFromLayerPairForPhotonConversion *>   Container;
  Container        theGenerators;

  OrderedHitPairs thePairs;

};
#endif
