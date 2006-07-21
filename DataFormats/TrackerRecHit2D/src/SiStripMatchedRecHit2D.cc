#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"


SiStripMatchedRecHit2D::SiStripMatchedRecHit2D( const LocalPoint& pos, const LocalError& err,
								const DetId& id , const SiStripRecHit2D* rMono,const SiStripRecHit2D* rStereo): BaseSiTrackerRecHit2DLocalPos(pos, err, id), componentMono_(*rMono),componentStereo_(*rStereo){}

