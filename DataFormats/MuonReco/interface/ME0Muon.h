#ifndef MuonReco_ME0Muon_h
#define MuonReco_ME0Muon_h
/** \class reco::ME0Muon ME0Muon.h DataFormats/MuonReco/interface/ME0Muon.h
 *  
 * \author David Nash NEU
 *
 *
 */
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

namespace reco {
 
  class ME0Muon : public RecoCandidate{
  public:
    ME0Muon();
    //ME0Muon( const TrackRef & t, const ME0Segment & s) { innerTrack_ = t; me0Segment_ = s;}
    ME0Muon( const TrackRef & t, const ME0Segment & s, const int v, const double c) { innerTrack_ = t; me0Segment_ = s; me0segid_=v; trackCharge_ = c;}
    virtual ~ME0Muon(){}     
    
    /// reference to Track reconstructed in the tracker only
    TrackRef innerTrack() const { return innerTrack_; }
    virtual TrackRef track() const { return innerTrack(); }
    /// set reference to Track
    void setInnerTrack( const TrackRef & t ) { innerTrack_ = t; }
    void setTrack( const TrackRef & t ) { setInnerTrack(t); }
    /// set reference to our new ME0Segment type
    void setME0Segment( const ME0Segment & s ) { me0Segment_ = s; }

    const ME0Segment& me0segment() const { return me0Segment_; }
    
    //Added for testing
    void setme0segid( const int v){me0segid_=v;}
    int me0segid() const {return me0segid_;}

    
    const GlobalPoint& globalTrackPosAtSurface() const { return globalTrackPosAtSurface_; }
    const GlobalVector& globalTrackMomAtSurface() const { return globalTrackMomAtSurface_; }
    const LocalPoint& localTrackPosAtSurface() const { return localTrackPosAtSurface_; }
    const LocalVector& localTrackMomAtSurface() const { return localTrackMomAtSurface_; }

    int trackCharge() const { return trackCharge_; }
    const AlgebraicSymMatrix66& globalTrackCov() const { return globalTrackCov_; }
    const AlgebraicSymMatrix55& localTrackCov() const { return localTrackCov_; }

    void setGlobalTrackPosAtSurface(const GlobalPoint& globalTrackPosAtSurface) { globalTrackPosAtSurface_ = globalTrackPosAtSurface; }
    void setGlobalTrackMomAtSurface(const GlobalVector& globalTrackMomAtSurface) { globalTrackMomAtSurface_ = globalTrackMomAtSurface; }
    void setLocalTrackPosAtSurface(const LocalPoint& localTrackPosAtSurface) { localTrackPosAtSurface_ = localTrackPosAtSurface; }
    void setLocalTrackMomAtSurface(const LocalVector& localTrackMomAtSurface) { localTrackMomAtSurface_ = localTrackMomAtSurface; }
    void setTrackCharge(const int& trackCharge) { trackCharge_ = trackCharge; }
    void setGlobalTrackCov(const AlgebraicSymMatrix66& trackCov) { globalTrackCov_ = trackCov; }
    void setLocalTrackCov(const AlgebraicSymMatrix55& trackCov) { localTrackCov_ = trackCov; }
     
  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;

    /// reference to Track reconstructed in the tracker only
    TrackRef innerTrack_;
    ME0Segment me0Segment_;
    int me0segid_;

    GlobalPoint globalTrackPosAtSurface_;
    GlobalVector globalTrackMomAtSurface_;

    LocalPoint localTrackPosAtSurface_;
    LocalVector localTrackMomAtSurface_;
    int trackCharge_;
    AlgebraicSymMatrix66 globalTrackCov_;
    AlgebraicSymMatrix55 localTrackCov_;

    //double xpull_,ypull_,xdiff_,ydiff_,phidirdiff_;
  };

}


#endif


