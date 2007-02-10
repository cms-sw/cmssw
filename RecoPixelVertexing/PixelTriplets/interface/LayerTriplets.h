#ifndef LayerTriplets_H
#define LayerTriplets_H

/** A class grouping pixel layers in pairs and associating a vector
    of layers to each layer pair. The layer pair is used to generate
    hit pairs and the associated vector of layers to generate
    a third hit confirming layer pair
 */

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
class DetLayer;
class LayerWithHits;

class LayerTriplets {
public:
  //  typedef pair<SeedLayerPairs::LayerPair, const LayerWithHits* > LayerTriplet;
  typedef std::pair<SeedLayerPairs::LayerPair, std::vector<const LayerWithHits*> > LayerPairAndLayers;
  LayerTriplets(){};
  virtual  ~LayerTriplets(){};
  //  virtual std::vector<LayerTriplet> operator()()= 0;
  virtual std::vector<LayerPairAndLayers> layers()= 0 ;
 
};

#endif

