#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

SiTrackerGSMatchedRecHit2D::SiTrackerGSMatchedRecHit2D( const LocalPoint& pos, const LocalError& err,
					  const DetId& id,
					  const int simhitId         ,
					  const int simtrackId       ,
					  const uint32_t eeId,
					  const int pixelMultiplicityX = -1,
					  const int pixelMultiplicityY = -1, 
					  const bool isMatched = false,
					  const SiTrackerGSRecHit2D* rMono = 0 , 
					  const SiTrackerGSRecHit2D* rStereo= 0 ):
  GSSiTrackerRecHit2DLocalPos(pos,err,id) ,
  simhitId_(simhitId) ,
  simtrackId_(simtrackId) ,
  eeId_(eeId) ,
  pixelMultiplicityAlpha_(pixelMultiplicityX), 
  pixelMultiplicityBeta_(pixelMultiplicityY), 
  isMatched_(isMatched), 
  componentMono_(*rMono), 
  componentStereo_(*rStereo)
{}

SiTrackerGSMatchedRecHit2D::SiTrackerGSMatchedRecHit2D( const LocalPoint& pos, const LocalError& err,
					  const DetId& id,
					  const int simhitId         ,
					  const int simtrackId       ,
					  const uint32_t eeId,
					  const int pixelMultiplicityX = -1,
							const int pixelMultiplicityY = -1):
  GSSiTrackerRecHit2DLocalPos(pos,err,id) ,
  simhitId_(simhitId) ,
  simtrackId_(simtrackId) ,
  eeId_(eeId) ,
  pixelMultiplicityAlpha_(pixelMultiplicityX), 
  pixelMultiplicityBeta_(pixelMultiplicityY), 
  isMatched_(0), 
  componentMono_(), 
  componentStereo_()
{}



