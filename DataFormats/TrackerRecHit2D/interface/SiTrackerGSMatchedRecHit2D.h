#ifndef SiTrackerGSMatchedRecHit2D_H
#define SiTrackerGSMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
class SiTrackerGSRecHit2D;

class SiTrackerGSMatchedRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
 public:
  
 SiTrackerGSMatchedRecHit2D()
   : GSSiTrackerRecHit2DLocalPos()
    , simtrackId_(-1)
    , isMatched_(false)
    , componentMono_()
    , componentStereo_(){}
  
  ~SiTrackerGSMatchedRecHit2D() {}
  
  SiTrackerGSMatchedRecHit2D( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet,
			      const uint32_t simtrackId)
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet)
    , simtrackId_(simtrackId)
    , isMatched_(false)
    , id_(-1)
    , eeId_(-1)
    {};

  SiTrackerGSMatchedRecHit2D( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet,
			      const uint32_t simtrackId,
			      const bool isMatched,
			      const SiTrackerGSRecHit2D & rMono, 
			      const SiTrackerGSRecHit2D & rStereo) 
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet)
    , simtrackId_(simtrackId)
    , isMatched_(isMatched)
    , componentMono_(rMono) 
    , componentStereo_(rStereo)
    , id_(-1)
    , eeId_(-1)
    {};


  virtual SiTrackerGSMatchedRecHit2D * clone() const {SiTrackerGSMatchedRecHit2D * p =  new SiTrackerGSMatchedRecHit2D( * this); p->load(); return p;}

  const uint32_t & id()          const { return id_;}
  const uint32_t & simtrackId()  const { return simtrackId_;}
  const uint32_t & eeId()   const { return eeId_;}
  const bool & isMatched()  const { return isMatched_;}
  const SiTrackerGSRecHit2D & monoHit() const { return componentMono_;}
  const SiTrackerGSRecHit2D & stereoHit() const { return componentStereo_;}

  void setId(uint32_t id){id_ = id;}
  void setEeId(uint32_t eeId){eeId_ = eeId;}

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
  
 private:
  
  const uint32_t simtrackId_;
  const bool isMatched_;
  
  const SiTrackerGSRecHit2D componentMono_;
  const SiTrackerGSRecHit2D componentStereo_;

  uint32_t id_;
  uint32_t eeId_;  
};



#endif
