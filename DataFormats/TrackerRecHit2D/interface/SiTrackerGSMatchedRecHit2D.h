#ifndef SiTrackerGSMatchedRecHit2D_H
#define SiTrackerGSMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
class SiTrackerGSRecHit2D;

class SiTrackerGSMatchedRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
 public:
  
 SiTrackerGSMatchedRecHit2D()
   : GSSiTrackerRecHit2DLocalPos()
    , isMatched_(false)
    , stereoHitFirst_(false)
    , componentMono_() 
    , componentStereo_()
    {}
  
  ~SiTrackerGSMatchedRecHit2D() {}
  
  SiTrackerGSMatchedRecHit2D( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet,
			      int32_t id)
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet,id,trackerHitRTTI::gsMatch)
    , isMatched_(false)
    , stereoHitFirst_(false)
    , componentMono_() 
    , componentStereo_()
    {};

  SiTrackerGSMatchedRecHit2D( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet,
			      int32_t id,
			      const bool isMatched,
			      const SiTrackerGSRecHit2D & rMono, 
			      const SiTrackerGSRecHit2D & rStereo) 
    : GSSiTrackerRecHit2DLocalPos(pos,err,idet,id,trackerHitRTTI::gsMatch)
    , isMatched_(isMatched)
    , stereoHitFirst_(false)
    , componentMono_(rMono) 
    , componentStereo_(rStereo)
    {};


  virtual SiTrackerGSMatchedRecHit2D * clone() const {SiTrackerGSMatchedRecHit2D * p =  new SiTrackerGSMatchedRecHit2D( * this); p->load(); return p;}

  const bool &                  isMatched()              const { return isMatched_;}
  const SiTrackerGSRecHit2D &   monoHit()                const { return componentMono_;}
  const SiTrackerGSRecHit2D &   stereoHit()              const { return componentStereo_;}
  const SiTrackerGSRecHit2D &   firstHit()               const { return stereoHitFirst_ ? componentStereo_ : componentMono_;}
  const SiTrackerGSRecHit2D &   secondHit()              const { return stereoHitFirst_ ? componentMono_ : componentStereo_;}
  void setStereoLayerFirst(bool stereoHitFirst = true){stereoHitFirst_ = stereoHitFirst;}

  void setHitCombinationId(int32_t hitCombinationId){
    GSSiTrackerRecHit2DLocalPos::setHitCombinationId(hitCombinationId);
    componentMono_.setHitCombinationId(hitCombinationId);
    componentStereo_.setHitCombinationId(hitCombinationId);
  }
  
 private:
  
  bool isMatched_;
  bool stereoHitFirst_;
  SiTrackerGSRecHit2D componentMono_;
  SiTrackerGSRecHit2D componentStereo_;
};


typedef SiTrackerGSMatchedRecHit2D                   FastTMRecHit; //Fast Tracker Matched RecHit
typedef std::vector<FastTMRecHit>                    FastTMRecHitCombination;
typedef std::vector<FastTMRecHitCombination>         FastTMRecHitCombinations;  

#endif
