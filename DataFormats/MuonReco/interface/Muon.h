#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
/** \class reco::Muon
 *  
 * A reconstructed Muon ment to be stored in the AOD
 * It contains a reference to a MuonExtra object
 * ment to be stored in the RECO
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Muon.h,v 1.13 2006/03/30 07:15:59 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonExtra.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

namespace reco {
 
  class Muon : public TrackBase {
  public:
    /// default constructor
    Muon() { }
    /// constructor from fit parameters and error matrix
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	  const Parameters &, const Covariance & );
    /// constructor from cartesian coordinates and covariance matrix.
    /// notice that the vertex passed must be 
    /// the point of closest approch to the beamline.    
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	  int q, const Point & v, const Vector & p, 
	  const PosMomError & err );
    /// set reference to Track reconstructed in the tracker only
    void setTrack( const TrackRef & ref ) { track_ = ref; }
    /// reference to Track reconstructed in the tracker only
    const TrackRef & track() const { return track_; }
    /// set reference to associated Ecal SuperCluster
    void setSuperCluster( const SuperClusterRef & ref ) { superCluster_ = ref; }
    /// reference to associated Ecal SuperCluster
    const SuperClusterRef & superCluster() const { return superCluster_; }

    /// reference to Track reconstructed in the muon detector only
    const TrackRef & standAloneMuon() const { return extra_->standAloneMuon(); }
    /// first iterator to RecHits
    trackingRecHit_iterator recHitsBegin() const { return extra_->recHitsBegin(); }
    /// last iterator to RecHits
    trackingRecHit_iterator recHitsEnd()   const { return extra_->recHitsEnd(); }
    /// number of RecHits
    size_t recHitsSize() const { return extra_->recHitsSize(); }
    /// set reference to "extra" object
    void setExtra( const MuonExtraRef & ref ) { extra_ = ref; }
    /// reference to "extra" object
    const MuonExtraRef & extra() const { return extra_; }

  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef track_;
    /// reference to associated Ecal SuperCluster
    SuperClusterRef superCluster_;
    /// reference to "extra" object
    MuonExtraRef extra_;
};

}

#endif
