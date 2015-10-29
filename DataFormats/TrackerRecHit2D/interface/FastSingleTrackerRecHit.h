#ifndef FastSingleTrackerRecHit_H
#define FastSingleTrackerRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "stdint.h"

class FastSingleTrackerRecHit : public FastTrackerRecHit {

    public:
    
    FastSingleTrackerRecHit ()
	: FastTrackerRecHit()
	, id_(-1)
	, eventId_(0) {}
    

    FastSingleTrackerRecHit (const LocalPoint& p, const LocalError&e, GeomDet const & idet,fastTrackerRecHitType::HitType hitType)
	: FastTrackerRecHit(p,e,idet,hitType)
	, id_(-1)
	, eventId_(0) {}
    
    public:

    virtual FastSingleTrackerRecHit * clone() const GCC11_OVERRIDE  {FastSingleTrackerRecHit * p =  new FastSingleTrackerRecHit( * this); p->load(); return p;}

    size_t                       nIds()                    const { return 1;}
    int32_t                      id(size_t i =0)           const { return i == 0 ? id_ : -1;}
    int32_t                      eventId(size_t i = 0)     const { return i == 0 ? eventId_ : -1;}
    size_t                       nSimTrackIds()            const { return simTrackIds_.size();}                             ///< see addSimTrackId(int32_t simTrackId)
    int32_t                      simTrackId(size_t i)      const { return i < simTrackIds_.size() ? simTrackIds_[i] : -1;}  ///< see addSimTrackId(int32_t simTrackId)
    int32_t                      simTrackEventId(size_t i) const { return i < simTrackIds_.size() ? eventId_ : -1;}  ///< see addSimTrackId(int32_t simTrackId)


    /// add an id number to the list of id numbers of SimTracks from which the hit originates
    /// the SimTrack id numbers are the indices of the SimTracks in the SimTrack collection
    void addSimTrackId(int32_t simTrackId)  {simTrackIds_.push_back(simTrackId);}
    
    /// set the hit id number
    /// for any particular hit in any particular event,
    // the hit id number must be unique within the list of hits in the event
    void setId(int32_t id)            {id_ = id;}

    /// set the hit's event number
    /// there is in principle no reason to play with this variable outside the PU mixing modules
    /// see Adjuster::doTheOffset(int bunchSpace, int bcr, TrackingRecHitCollection & trackingrechits, unsigned int evtNr, int vertexOffset)
    void setEventId(int32_t eventId)    {eventId_ = eventId;}

    private:

    int32_t id_;                                             ///< see setId(int32_t id)
    int32_t eventId_;                                        ///< see setEeId(int32_t eeid)
    std::vector<int32_t> simTrackIds_;                       ///< see addSimTrackIds(int32_t)
    
};

#endif
