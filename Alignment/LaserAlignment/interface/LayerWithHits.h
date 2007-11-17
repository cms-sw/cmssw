#ifndef LaserAlignment_LayerWithHits_H
#define LaserAlignment_LayerWithHits_H

/** \class LayerWithHits
 *  find layers with hits for Laser Tracks
 *
 *  $Date: Thu May 10 08:37:14 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */
 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

class LayerWithHits
{
 public:
  LayerWithHits(const DetLayer *dl,std::vector<const TrackingRecHit*> theInputHits):
    theDetLayer(dl),theHits(theInputHits){}
  
  LayerWithHits( const DetLayer *dl,const SiPixelRecHitCollection::range range);
  LayerWithHits( const DetLayer *dl,const SiStripRecHit2DCollection::range range);
  LayerWithHits( const DetLayer *dl,const SiStripMatchedRecHit2DCollection::range range);

  //destructor
  ~LayerWithHits(){}
  
  /// return the recHits of the Layer
  const std::vector<const TrackingRecHit*>& recHits() const {return theHits;}

  //detlayer
  const  DetLayer* layer()  const {return theDetLayer;}
  
 private:
  const DetLayer* theDetLayer;
  std::vector<const TrackingRecHit*> theHits;
};
#endif
