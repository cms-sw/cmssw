#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"


SiTrackerGSRecHit2D::SiTrackerGSRecHit2D( const LocalPoint& pos, const LocalError& err,
					  const DetId& id,
					  const int simhitId         ,
					  const int simtrackId       ,
					  const uint32_t eeId        ,
					  ClusterRef const&  cluster ,
					  const int pixelMultiplicityX = -1,
					  const int pixelMultiplicityY = -1 
					   ): 
  GSSiTrackerRecHit2DLocalPos(pos,err,id) ,
  simhitId_(simhitId) ,
  simtrackId_(simtrackId) ,
  eeId_(eeId) ,
  cluster_(cluster), 
  pixelMultiplicityAlpha_(pixelMultiplicityX), 
  pixelMultiplicityBeta_(pixelMultiplicityY){}
