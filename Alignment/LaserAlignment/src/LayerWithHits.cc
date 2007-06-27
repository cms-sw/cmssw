/** \file LayerWithHits.cc
*  find layer with hits for Laser Tracks
  *
  *  $Date: 2007/05/10 06:41:10 $
  *  $Revision: 1.1 $
  *  \author Maarten Thomas
 */

 #include "Alignment/LaserAlignment/interface/LayerWithHits.h"

LayerWithHits::LayerWithHits( const DetLayer *dl,const SiPixelRecHitCollection::range range){
  theDetLayer = dl;
  for(SiPixelRecHitCollection::const_iterator it = range.first; it != range.second; it++){
    theHits.push_back( &(*it) );
  }
}

LayerWithHits::LayerWithHits( const DetLayer *dl,const SiStripRecHit2DCollection::range range){
  theDetLayer = dl;
  for(SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; it++){
    theHits.push_back( &(*it) );
  }
}

LayerWithHits::LayerWithHits( const DetLayer *dl,const SiStripMatchedRecHit2DCollection::range range){
  theDetLayer = dl;
  for(SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; it++){
    theHits.push_back( &(*it) );
  }
}
