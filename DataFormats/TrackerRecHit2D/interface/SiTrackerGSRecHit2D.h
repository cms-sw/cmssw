#ifndef SiTrackerGSRecHit2D_H
#define SiTrackerGSRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FastSimDataFormats/External/interface/FastTrackerClusterCollection.h" 

class SiTrackerGSRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
 public:
  
  
  
 SiTrackerGSRecHit2D()
   : GSSiTrackerRecHit2DLocalPos()
    , id_(-1)
    , eeId_(-1)
    , hitCombinationId_(-1)
    {}
  
  ~SiTrackerGSRecHit2D() {}
  
 SiTrackerGSRecHit2D( const LocalPoint & pos,
		      const LocalError & err,
		      const GeomDet & idet)
   : GSSiTrackerRecHit2DLocalPos(pos,err,idet)
    , id_(-1)
    , eeId_(-1)
    , hitCombinationId_(-1)
    {}
  
  virtual SiTrackerGSRecHit2D * clone() const {SiTrackerGSRecHit2D * p = new SiTrackerGSRecHit2D( * this); p->load(); return p;}
  
  int32_t                      id()                     const { return id_;}
  int32_t                      eeId()                   const { return eeId_;}
  const std::vector<int32_t> & simTrackIds()            const { return simTrackIds_;}
  size_t                       nSimTrackIds()           const { return simTrackIds_.size();}
  int32_t                      simTrackId(size_t index) const { return index < simTrackIds_.size() ? simTrackIds_[index] : -1;}
  
  void setId(int32_t id){id_ = id;}
  void setEeId(int32_t eeId){eeId_ = eeId;}
  void setHitCombinationId(int32_t hitCombinationId) {hitCombinationId_ = hitCombinationId;}
  void addSimTrackId(int32_t simTrackId)  {simTrackIds_.push_back(simTrackId);}
  void addSimTrackIds(const std::vector<int32_t> & simTrackIds)  {simTrackIds_.insert(simTrackIds_.end(),simTrackIds.begin(),simTrackIds.end());}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
 private:
  
  int32_t id_;
  int32_t eeId_;
  int32_t hitCombinationId_;
  std::vector<int32_t> simTrackIds_;
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
