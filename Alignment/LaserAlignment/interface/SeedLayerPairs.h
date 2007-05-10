#ifndef LaserAlignment_SeedLayerPairs_H
#define LaserAlignment_SeedLayerPairs_H

/** \class SeedLayerPairs
 *  interface to access pairs of layers; used for seedgenerator
 *
 *  $Date: Thu May 10 13:54:16 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */
 
#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
class DetLayer;
class LayerWithHits;

class SeedLayerPairs {
public:

  typedef std::pair< const LayerWithHits*, const LayerWithHits*>        LayerPair;

  SeedLayerPairs() {};
  virtual ~SeedLayerPairs() {};
    virtual std::vector<LayerPair> operator()()= 0;
  

};

#endif
