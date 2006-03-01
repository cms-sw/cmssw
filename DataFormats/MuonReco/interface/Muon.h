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
 * \version $Id$
 *
 */
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"
#include"DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include"DataFormats/TrackReco/interface/TrackExtension.h"
#include"DataFormats/MuonReco/interface/MuonFwd.h"
#include"DataFormats/MuonReco/interface/MuonExtraFwd.h"

namespace reco {
 
  class Muon : public TrackBase, public TrackExtension<MuonExtraRef> {
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
    /// reference to Track reconstructed in the muon detector only
    const TrackRef & standAloneMuon() const;

  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef track_;
    /// reference to "extra" object
    MuonExtraRef extra_;
};

}

#endif
