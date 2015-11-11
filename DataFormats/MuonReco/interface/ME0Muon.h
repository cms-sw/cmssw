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

#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

namespace reco {
 
  class ME0Muon {
  public:
    ME0Muon();
    //ME0Muon( const TrackRef & t, const ME0Segment & s) { innerTrack_ = t; me0Segment_ = s;}
    ME0Muon( const TrackRef & t, const ME0Segment & s, const int v) { innerTrack_ = t; me0Segment_ = s; me0segid_=v;}
    virtual ~ME0Muon(){}     
    
    /// reference to Track reconstructed in the tracker only
    virtual TrackRef innerTrack() const { return innerTrack_; }
    virtual TrackRef track() const { return innerTrack(); }
    /// set reference to Track
    virtual void setInnerTrack( const TrackRef & t ) { innerTrack_ = t; }
    virtual void setTrack( const TrackRef & t ) { setInnerTrack(t); }
    /// set reference to our new ME0Segment type
    virtual void setME0Segment( const ME0Segment & s ) { me0Segment_ = s; }

    virtual ME0Segment me0segment() const { return me0Segment_; }
    
    //Added for testing
    virtual void setme0segid( const int v){me0segid_=v;}
    virtual int me0segid() const {return me0segid_;}

    /// a bunch of useful accessors
    int charge() const { return innerTrack_.get()->charge(); }
    /// polar angle  
    double theta() const { return innerTrack_.get()->theta(); }
    /// momentum vector magnitude
    double p() const { return innerTrack_.get()->p(); }
    /// track transverse momentum
    double pt() const { return innerTrack_.get()->pt(); }
    /// x coordinate of momentum vector
    double px() const { return innerTrack_.get()->px(); }
    /// y coordinate of momentum vector
    double py() const { return innerTrack_.get()->py(); }
    /// z coordinate of momentum vector
    double pz() const { return innerTrack_.get()->pz(); }
    /// azimuthal angle of momentum vector
    double phi() const { return innerTrack_.get()->phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return innerTrack_.get()->eta(); }
     
  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef innerTrack_;
    ME0Segment me0Segment_;
    int me0segid_;
  };

}


#endif


