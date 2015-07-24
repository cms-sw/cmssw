#ifndef GSSiTrackerRecHit2DLocalPos_H
#define GSSiTrackerRecHit2DLocalPos_H

#include "BaseTrackerRecHit.h"
#include "stdint.h"

class GSSiTrackerRecHit2DLocalPos : public BaseTrackerRecHit {
public:

  GSSiTrackerRecHit2DLocalPos()
    : BaseTrackerRecHit()
    , id_(-1)
    , eeId_(-1)
    , hitCombinationId_(-1)
    {}

  ~GSSiTrackerRecHit2DLocalPos() {}

  GSSiTrackerRecHit2DLocalPos( const LocalPoint& p, const LocalError&e, GeomDet const & idet,int32_t id,trackerHitRTTI::RTTI rt) 
    : BaseTrackerRecHit(p,e,idet,rt) 
    , id_(id)
    , eeId_(0)
    , hitCombinationId_(-1)
  {store();}

  virtual GSSiTrackerRecHit2DLocalPos * clone() const =0;

  
  virtual void getKfComponents( KfComponentsHolder & holder ) const {
     getKfComponents2D(holder);
  }

  virtual int dimension() const { return 2;}

  virtual std::vector<const TrackingRecHit*> recHits() const { return std::vector<TrackingRecHit const*>();}
  virtual std::vector<TrackingRecHit*> recHits()  { return std::vector<TrackingRecHit*>();}

  // shall I support FakeCluster?
  virtual OmniClusterRef const & firstClusterRef() const;

  int32_t                      id()                     const { return id_;}
  int32_t                      eeId()                   const { return eeId_;}
  int32_t                      hitCombinationId()       const { return hitCombinationId_;}
  const std::vector<int32_t> & simTrackIds()            const { return simTrackIds_;}
  size_t                        nSimTrackIds()           const { return simTrackIds_.size();}
  int32_t                       simTrackId(size_t index) const { return index < simTrackIds_.size() ? simTrackIds_[index] : -1;}

  void setId(int32_t id)            {id_ = id;}
  void setEeId(int32_t eeId)        {eeId_ = eeId;}
  virtual void setHitCombinationId(int32_t hitCombinationId) {hitCombinationId_ = hitCombinationId;}
  void addSimTrackId(int32_t simTrackId)  {simTrackIds_.push_back(simTrackId);}
  void addSimTrackIds(const std::vector<int32_t> & simTrackIds)  {simTrackIds_.insert(simTrackIds_.end(),simTrackIds.begin(),simTrackIds.end());}

  bool sharesInput( const TrackingRecHit* other, SharedInputType what) const {
    const GSSiTrackerRecHit2DLocalPos * other_casted = static_cast<const GSSiTrackerRecHit2DLocalPos *>(other);
    return this->id_ == other_casted->id_ && this->getRTTI() == other_casted->getRTTI() &&  this->eeId_ == other_casted->eeId_;
  }
  
private:
  virtual TrackingRecHit* clone(const TkCloner&, const TrajectoryStateOnSurface&) const { return clone();}
  std::vector<int32_t> simTrackIds_;

protected:

  int32_t id_;
  int32_t eeId_;
  int32_t hitCombinationId_;

  LocalPoint m_myPos;
  LocalError m_myErr;

  void store() { m_myPos=pos_;  m_myErr=err_;}  
  void load()  { pos_=m_myPos; err_=m_myErr;}

  
};

// Comparison operators
inline bool operator<( const GSSiTrackerRecHit2DLocalPos& one, const GSSiTrackerRecHit2DLocalPos& other) {
  return ( one.geographicalId() < other.geographicalId() );
}

#endif
