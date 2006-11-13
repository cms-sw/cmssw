#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"


SiTrackerGSRecHit2D::SiTrackerGSRecHit2D( const LocalPoint& pos, const LocalError& err,
					  const DetId& id,
					  const int simhitId           = -1,
					  const int simtrackId         = -1,
					  const int pixelMultiplicityX = -1,
					  const int pixelMultiplicityY = -1 ):
  BaseSiTrackerRecHit2DLocalPos(pos,err,id) ,
  simhitId_(simhitId) ,
  simtrackId_(simtrackId) ,
  pixelMultiplicityAlpha_(pixelMultiplicityX), 
  pixelMultiplicityBeta_(pixelMultiplicityY) {}
