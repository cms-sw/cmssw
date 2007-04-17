#ifndef PixelLayerTriplets_H
#define PixelLayerTriplets_H

/** A class grouping pixel layers in pairs and associating a vector
    of layers to each layer pair. The layer pair is used to generate
    hit pairs and the associated vector of layers to generate
    a third hit confirming layer pair
 */

#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include <vector>
class BarrelDetLayer;
class ForwardDetLayer;

using namespace std;

class PixelLayerTriplets {
public:
  PixelLayerTriplets();
  ~PixelLayerTriplets();
  typedef PixelSeedLayerPairs::LayerPair LayerPair;
  typedef pair<LayerPair, vector<const LayerWithHits*> > LayerPairAndLayers;
  vector<LayerPairAndLayers> layers() ;
  void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup);

private:

  SiPixelRecHitCollection::range map_range1;
  SiPixelRecHitCollection::range map_range2;
  SiPixelRecHitCollection::range map_range3;

  SiPixelRecHitCollection::range map_diskneg1;
  SiPixelRecHitCollection::range map_diskneg2;

  SiPixelRecHitCollection::range map_diskpos1;
  SiPixelRecHitCollection::range map_diskpos2;

  TrackerLayerIdAccessor acc;
  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;

  LayerWithHits *pos1;
  LayerWithHits *pos2;

  LayerWithHits *neg1;
  LayerWithHits *neg2;

  vector<BarrelDetLayer*> bl;
  vector<ForwardDetLayer*> fpos;
  vector<ForwardDetLayer*> fneg;
};

#endif

