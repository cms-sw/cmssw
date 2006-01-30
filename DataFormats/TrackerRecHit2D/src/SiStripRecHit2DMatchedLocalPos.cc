#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPos.h"


SiStripRecHit2DMatchedLocalPos::SiStripRecHit2DMatchedLocalPos( const LocalPoint& pos, const LocalError& err,
								const DetId& id , const SiStripRecHit2DLocalPos* rMono,const SiStripRecHit2DLocalPos* rStereo): BaseSiStripRecHit2DLocalPos(pos, err, id), componentMono_(rMono),componentStereo_(rStereo){}

