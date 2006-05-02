#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
/** \class reco::Muon Muon.h DataFormats/MuonReco/interface/Muon.h
 *  
 * A reconstructed Muon.
 * contains reference to three fits:
 *  - tracker alone
 *  - muon detector alone
 *  - combined muon plus tracker
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Muon.h,v 1.16 2006/04/26 07:16:44 llista Exp $
 *
 */
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/RecoCandidate.h"

namespace reco {
 
  class Muon : public RecoCandidate {
  public:
    Muon() { }
    /// constructor from values
    Muon(  Charge, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
    /// reference to Track reconstructed in the tracker only
    const TrackRef & track() const { return track_; }
    /// reference to Track reconstructed in the muon detector only
    const TrackRef & standAlone() const { return standAlone_; }
    /// reference to Track reconstructed in both tracked and muon detector
    const TrackRef & combined() const { return combined_; }
    /// reference to associated Ecal SuperCluster
    const SuperClusterRef & superCluster() const { return superCluster_; }
    /// set reference to Track
    void setTrack( const TrackRef & t ) { track_ = t; }
    /// set reference to Track
    void setStandAlone( const TrackRef & t ) { standAlone_ = t; }
    /// set reference to Track
    void setCombined( const TrackRef & t ) { combined_ = t; }
    /// set reference to associated Ecal SuperCluster
    void setSuperCluster( const SuperClusterRef & ref ) { superCluster_ = ref; }

  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef track_;
    /// reference to Track reconstructed in the muon detector only
    TrackRef standAlone_;
    /// reference to Track reconstructed in both tracked and muon detector
    TrackRef combined_;
    /// reference to associated Ecal SuperCluster
    SuperClusterRef superCluster_;
};

}

#endif
