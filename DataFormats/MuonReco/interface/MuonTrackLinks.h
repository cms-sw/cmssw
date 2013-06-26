#ifndef DataFormats_MuonReco_MuonTrackLinks_H
#define DataFormats_MuonReco_MuonTrackLinks_H

/** \class MuonTrackLinks
 *  Transient format to keep the links between the three different tracks
 *  which are built in the RecoMuon tracking code.
 *  This data format is meant to be used internally only.
 *
 *  $Date: 2007/05/04 18:25:27 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class MuonTrackLinks {

  public:

    /// Default Constructor
    MuonTrackLinks(){}
    
    /// Constructor
    MuonTrackLinks(reco::TrackRef tk, reco::TrackRef sta, reco::TrackRef glb):
      theTkTrack(tk),theStaTrack(sta),theGlbTrack(glb){}

    /// Destructor
    virtual ~MuonTrackLinks(){};

    // Operations
  
    /// get the tracker's track which match with the stand alone muon tracks
    inline reco::TrackRef trackerTrack() const {return theTkTrack;}

    /// get the track built with the muon spectrometer alone
    inline reco::TrackRef standAloneTrack() const {return theStaTrack;}

    /// get the combined track
    inline reco::TrackRef globalTrack() const {return theGlbTrack;}

    /// set the ref to tracker's track
    inline void setTrackerTrack(reco::TrackRef tk) {theTkTrack = tk;}

    /// set the ref to stand alone track
    inline void setStandAloneTrack(reco::TrackRef sta) {theStaTrack = sta;}

    /// set the ref to combined track
    inline void setGlobalTrack(reco::TrackRef glb) {theGlbTrack = glb;}
  
  protected:

  private:
    
    /// ref to tracker's track which match with the stand alone muon tracks
    reco::TrackRef theTkTrack;

    /// ref to the track built with the muon spectrometer alone
    reco::TrackRef theStaTrack;

    /// ref to the combined track
    reco::TrackRef theGlbTrack;
  };
}
#endif

