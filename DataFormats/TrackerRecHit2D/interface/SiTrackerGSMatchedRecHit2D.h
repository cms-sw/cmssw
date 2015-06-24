#ifndef SiTrackerGSMatchedRecHit2D_H
#define SiTrackerGSMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
class SiTrackerGSRecHit2D;

class SiTrackerGSMatchedRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
 public:
  
 SiTrackerGSMatchedRecHit2D()
   : GSSiTrackerRecHit2DLocalPos()
    , id_(-1)
    , eeId_(-1)
    , hitCombinationId_(-1)
    , isMatched_(false)
    , componentMono_() 
    , componentStereo_()
    {}
  
  ~SiTrackerGSMatchedRecHit2D() {}
  
  SiTrackerGSMatchedRecHit2D( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet)
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet)
    , id_(-1)
    , eeId_(-1)
    , hitCombinationId_(-1)
    , isMatched_(false)
    , componentMono_() 
    , componentStereo_()
    {};

  SiTrackerGSMatchedRecHit2D( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet,
			      const bool isMatched,
			      const SiTrackerGSRecHit2D & rMono, 
			      const SiTrackerGSRecHit2D & rStereo) 
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet)
    , id_(-1)
    , eeId_(-1)
    , hitCombinationId_(-1)
    , isMatched_(isMatched)
    , componentMono_(rMono) 
    , componentStereo_(rStereo)
    {};


  virtual SiTrackerGSMatchedRecHit2D * clone() const {SiTrackerGSMatchedRecHit2D * p =  new SiTrackerGSMatchedRecHit2D( * this); p->load(); return p;}

  int32_t                      id()                     const { return id_;}
  int32_t                      eeId()                   const { return eeId_;}
  int32_t                      hitCombinationId()       const { return hitCombinationId_;}
  const std::vector<int32_t> & simTrackIds()            const { return simTrackIds_;}
  size_t                        nSimTrackIds()           const { return simTrackIds_.size();}
  int32_t                       simTrackId(size_t index) const { return index < simTrackIds_.size() ? simTrackIds_[index] : -1;}
  const bool &                  isMatched()              const { return isMatched_;}
  const SiTrackerGSRecHit2D &   monoHit()                const { return componentMono_;}
  const SiTrackerGSRecHit2D &   stereoHit()              const { return componentStereo_;}

  void setId(int32_t id)            {id_ = id;}
  void setEeId(int32_t eeId)        {eeId_ = eeId;}
  void setHitCombinationId(int32_t hitCombinationId) {hitCombinationId_ = hitCombinationId;}
  void addSimTrackId(int32_t simTrackId)  {simTrackIds_.push_back(simTrackId);}
  void addSimTrackIds(const std::vector<int32_t> & simTrackIds)  {simTrackIds_.insert(simTrackIds_.end(),simTrackIds.begin(),simTrackIds.end());}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
 private:
  
  int32_t id_;
  int32_t eeId_;
  int32_t hitCombinationId_;
  std::vector<int32_t> simTrackIds_;

  bool isMatched_;
  SiTrackerGSRecHit2D componentMono_;
  SiTrackerGSRecHit2D componentStereo_;
};


typedef SiTrackerGSMatchedRecHit2D                   FastTMRecHit; //Fast Tracker Matched RecHit
typedef std::vector<FastTMRecHit>                    FastTMRecHitCombination;
typedef std::vector<FastTMRecHitCombination>         FastTMRecHitCombinations;  

#endif
