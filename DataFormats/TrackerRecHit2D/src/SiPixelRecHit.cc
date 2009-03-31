#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

SiPixelRecHit::SiPixelRecHit( const LocalPoint& pos, const LocalError& err,
			      const DetId& id,
			      SiPixelClusterRefNew const&  cluster): 
  BaseSiTrackerRecHit2DLocalPos(pos,err,id),
  cluster_(cluster) 

{
}

bool SiPixelRecHit::sharesInput( const TrackingRecHit* other, 
				 SharedInputType what) const
{
  if (geographicalId() != other->geographicalId()) return false;
  if(! other->isValid()) return false;

  const SiPixelRecHit* otherCast = static_cast<const SiPixelRecHit*>(other);

  return cluster_ == otherCast->cluster();
}


//--- The overall probability.  flags is the 32-bit-packed set of flags that
//--- our own concrete implementation of clusterProbability() uses to direct
//--- the computation based on the information stored in the quality word
//--- (and which was computed by the CPE).  The default of flags==0 returns
//--- probX*probY*(1-log(probX*probY)) because of Morris' note.
//--- Flags are static and kept in the transient rec hit.
float SiPixelRecHit::clusterProbability(unsigned int flags) const
{
	if (!hasFilledProb()) {
		return 1;
	}
	else if (flags == 0) {
		if(probabilityX() != 0 && probabilityY() != 0) {
			return probabilityX()*probabilityY() * (1 - log(probabilityX()*probabilityY()));
		}
		else {
			return 0;
		}
	}
	else if (flags == 1) {
    return probabilityX();
	}
	else if (flags == 2) {
		return probabilityY();
  }
	else {
		return 0;
	}
	
}

