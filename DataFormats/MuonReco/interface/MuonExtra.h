#ifndef MuonReco_MuonExtra_h
#define MuonReco_MuonExtra_h
//
// $Id: TrackExtra.h,v 1.4 2006/02/16 14:24:40 llista Exp $
//
// Definition of TrackExtra class for RECO
//
// Author: Luca Lista
//
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/MuonReco/interface/MuonExtraFwd.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class MuonExtra : public TrackExtra {
  public:
    MuonExtra() { }
    MuonExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok );

    const TrackRef & trackerSegment() const { return trackerSegment_; }
    void setTrackerSegment( const TrackRef & ref ) { trackerSegment_ = ref; }
    const TrackRef & muonSegment() const { return muonSegment_; }
    void setMuonSegment( const TrackRef & ref ) { muonSegment_ = ref; }

  private:
    TrackRef trackerSegment_, muonSegment_;
  };

}

#endif
