#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"


SiTrackerGSRecHit2D::SiTrackerGSRecHit2D( const LocalPoint& pos, const LocalError& err,
					  const DetId& id,
					  const int&          simhitId,
					  const unsigned int& simtrackId,
					  const unsigned int& pixelMultiplicityX,
					  const unsigned int& pixelMultiplicityY ):
  BaseSiTrackerRecHit2DLocalPos(pos,err,id) ,
  simhitId_(simhitId) ,
  simtrackId_ (simtrackId) ,
  pixelMultiplicityAlpha_ (pixelMultiplicityX), 
  pixelMultiplicityBeta_  (pixelMultiplicityY) {}
