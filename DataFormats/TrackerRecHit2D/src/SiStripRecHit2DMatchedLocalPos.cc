#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"


SiStripRecHit2DMatchedLocalPos::SiStripRecHit2DMatchedLocalPos( const LocalPoint& pos, const LocalError& err,
								const DetId& id , const SiStripRecHit2DLocalPos* rStereo, const SiStripRecHit2DLocalPos* rMono): BaseSiStripRecHit2DLocalPos(pos, err, id), componentStereo_(rStereo), componentMono_(rMono){}
//const DetId& id , const SiStripRecHit2DLocalPos* rStereo, const SiStripRecHit2DLocalPos* rMono): BaseSiStripRecHit2DLocalPos(pos, err, id){}
