#ifndef PixelLayerTriplets_H
#define PixelLayerTriplets_H

/** A class grouping pixel layers in pairs and associating a vector
    of layers to each layer pair. The layer pair is used to generate
    hit pairs and the associated vector of layers to generate
    a third hit confirming layer pair
 */

class BarrelDetLayer;
class ForwardDetLayer;
namespace edm { class Event; }
namespace edm { class EventSetup; }

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <vector>

class PixelLayerTriplets {
public:
  PixelLayerTriplets();
  ~PixelLayerTriplets();
  typedef PixelSeedLayerPairs::LayerPair LayerPair;
  typedef std::pair<LayerPair, std::vector<const LayerWithHits*> > LayerPairAndLayers;
  std::vector<LayerPairAndLayers> layers() ;
  void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup);
  void init(const edm::Event& ev, const edm::EventSetup& es);

private:
  TrackerLayerIdAccessor acc;
  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;
  LayerWithHits *pos1;
  LayerWithHits *pos2;
  LayerWithHits *neg1;
  LayerWithHits *neg2;
  LayerWithHits *tib1;
};

#endif

