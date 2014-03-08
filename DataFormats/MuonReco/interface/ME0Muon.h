#ifndef MuonReco_ME0Muon_h
#define MuonReco_ME0Muon_h
/** \class reco::ME0Muon ME0Muon.h DataFormats/MuonReco/interface/ME0Muon.h
 *  
 * A lightweight reconstructed Muon to store low momentum muons without matches
 * in the muon detectors. Contains:
 *  - reference to a silicon tracker track
 *  - calorimeter energy deposition
 *  - calo compatibility variable
 *
 * \author Dmytro Kovalskyi, UCSB
 *
 * \version $Id: ME0Muon.h,v 1.4 2009/03/15 03:33:32 dmytro Exp $
 *
 */
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <DataFormats/MuonReco/interface/EmulatedME0SegmentCollection.h>

namespace reco {
 
  class ME0Muon {
  public:
    ME0Muon();
    ME0Muon( const TrackRef & t, const EmulatedME0SegmentRef & s) { innerTrack_ = t; me0Segment_ = s;}
    virtual ~ME0Muon(){}     
    
    /// reference to Track reconstructed in the tracker only
    virtual TrackRef innerTrack() const { return innerTrack_; }
    virtual TrackRef track() const { return innerTrack(); }
    /// set reference to Track
    virtual void setInnerTrack( const TrackRef & t ) { innerTrack_ = t; }
    virtual void setTrack( const TrackRef & t ) { setInnerTrack(t); }
    /// set reference to our new EmulatedME0Segment type
    virtual void setEmulatedME0Segment( const EmulatedME0SegmentRef & s ) { me0Segment_ = s; }

    virtual EmulatedME0SegmentRef me0segment() const { return me0Segment_; }

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
    EmulatedME0SegmentRef me0Segment_;
  };

}


#endif


