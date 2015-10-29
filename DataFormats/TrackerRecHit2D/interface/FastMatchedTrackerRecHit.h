#ifndef FastMatchedTrackerRecHit_H
#define FastMatchedTrackerRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

class FastMatchedTrackerRecHit : public FastTrackerRecHit{
    
    public:
    
    FastMatchedTrackerRecHit()
	: stereoHitFirst_(false)
	{}
    
    ~FastMatchedTrackerRecHit() {}
    
    FastMatchedTrackerRecHit( const LocalPoint & pos, 
			      const LocalError & err,
			      const GeomDet & idet,
			      const FastSingleTrackerRecHit & rMono, 
			      const FastSingleTrackerRecHit & rStereo,
			      bool stereoHitFirst = false) 
	: FastTrackerRecHit(pos,err,idet,fastTrackerRecHitType::siStripMatched2D)
	, stereoHitFirst_(stereoHitFirst)
	, componentMono_(rMono) 
	, componentStereo_(rStereo)
    {};
    
    virtual FastMatchedTrackerRecHit * clone() const {FastMatchedTrackerRecHit * p =  new FastMatchedTrackerRecHit( * this); p->load(); return p;}

    size_t    nIds()                    const { return 2;}
    int32_t   id(size_t i = 0)          const { return i==0 ? monoHit().id() : stereoHit().id(); }
    int32_t   eventId(size_t i = 0)     const { return i==0 ? monoHit().eventId() : stereoHit().eventId(); }
    
    size_t    nSimTrackIds()            const { return componentMono_.nSimTrackIds() + componentStereo_.nSimTrackIds();}                             ///< see addSimTrackId(int32_t simTrackId)
    int32_t   simTrackId(size_t i)      const { return i < componentMono_.nSimTrackIds() ? componentMono_.simTrackId(i) : componentStereo_.simTrackId(i-componentMono_.nSimTrackIds()); }
    int32_t   simTrackEventId(size_t i) const { return i < componentMono_.nSimTrackIds() ? componentMono_.simTrackEventId(i) : componentStereo_.simTrackEventId(i-componentMono_.nSimTrackIds()); }
    
    const FastSingleTrackerRecHit &   monoHit()                const { return componentMono_;}
    const FastSingleTrackerRecHit &   stereoHit()              const { return componentStereo_;}
    const FastSingleTrackerRecHit &   firstHit()               const { return stereoHitFirst_ ? componentStereo_ : componentMono_;}
    const FastSingleTrackerRecHit &   secondHit()              const { return stereoHitFirst_ ? componentMono_ : componentStereo_;}
    

    void setStereoLayerFirst(bool stereoHitFirst = true){stereoHitFirst_ = stereoHitFirst;}
    void setEventId(int32_t eventId){componentMono_.setEventId(eventId);componentStereo_.setEventId(eventId);}

    void setRecHitCombinationIndex(int32_t recHitCombinationIndex) {
	FastTrackerRecHit::setRecHitCombinationIndex(recHitCombinationIndex);
	componentMono_.setRecHitCombinationIndex(recHitCombinationIndex);
	componentStereo_.setRecHitCombinationIndex(recHitCombinationIndex);
    }

    private:
  
    bool stereoHitFirst_;
    FastSingleTrackerRecHit componentMono_;
    FastSingleTrackerRecHit componentStereo_;
};

#endif
