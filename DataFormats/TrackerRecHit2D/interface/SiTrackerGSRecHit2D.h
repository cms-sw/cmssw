#ifndef SiTrackerGSRecHit2D_H
#define SiTrackerGSRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FastSimDataFormats/External/interface/FastTrackerClusterCollection.h" 

class SiTrackerGSRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
 public:
  
  
  
 SiTrackerGSRecHit2D()
   : GSSiTrackerRecHit2DLocalPos()
    , simtrackId_() {}
  
  ~SiTrackerGSRecHit2D() {}
  
  SiTrackerGSRecHit2D( const LocalPoint & pos, 
		       const LocalError & err,
		       const GeomDet & idet,
		       const uint32_t simtrackId)
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet)
    , simtrackId_(simtrackId)
    , id_(-1)
    , eeId_(-1)
    {};
  
  virtual SiTrackerGSRecHit2D * clone() const {SiTrackerGSRecHit2D * p = new SiTrackerGSRecHit2D( * this); p->load(); return p;}
  
  const uint32_t & id()          const { return id_;}
  const uint32_t& simtrackId()  const { return simtrackId_;}
  const uint32_t& eeId()   const { return eeId_;}
  
  void setId(uint32_t id){id_ = id;}
  void setEeId(uint32_t eeId){eeId_ = eeId;}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
 private:
  
  const uint32_t simtrackId_;
  uint32_t id_;
  uint32_t eeId_;
  
};

// Comparison operators
inline bool operator<( const SiTrackerGSRecHit2D& one, const SiTrackerGSRecHit2D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
