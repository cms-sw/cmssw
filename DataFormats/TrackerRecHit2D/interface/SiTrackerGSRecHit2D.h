#ifndef SiTrackerGSRecHit2D_H
#define SiTrackerGSRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FastSimDataFormats/External/interface/FastTrackerClusterCollection.h" 

class SiTrackerGSRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
 public:
  
 SiTrackerGSRecHit2D()
   : GSSiTrackerRecHit2DLocalPos()
    {}
  
  ~SiTrackerGSRecHit2D() {}
  
 SiTrackerGSRecHit2D( const LocalPoint & pos,
		      const LocalError & err,
		      const GeomDet & idet,
		      int32_t id)
   : GSSiTrackerRecHit2DLocalPos(pos,err,idet,id,trackerHitRTTI::gs)
    {}
  
  virtual SiTrackerGSRecHit2D * clone() const {SiTrackerGSRecHit2D * p = new SiTrackerGSRecHit2D( * this); p->load(); return p;}
  
 private:
  
};

// Comparison operators
inline bool operator<( const SiTrackerGSRecHit2D& one, const SiTrackerGSRecHit2D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

typedef SiTrackerGSRecHit2D                   FastTRecHit; //Fast Tracker RecHit

#endif
