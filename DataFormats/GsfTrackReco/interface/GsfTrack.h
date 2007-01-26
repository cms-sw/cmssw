#ifndef GsfTrackReco_GsfTrack_h
#define GsfTrackReco_GsfTrack_h
/** Extension of reco::Track for GSF. It contains
 * one additional Ref to a GsfTrackExtra object.
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"

namespace reco {

  class GsfTrack : public Track {
  public:
    /// default constructor
    GsfTrack() { }
    /// constructor from fit parameters and error matrix
    /// notice that the reference point must be 
    /// the point of closest approch to the beamline.    
    GsfTrack( double chi2, double ndof, const Point &, const Vector &, int charge,
	      const CovarianceMatrix & );
    /// set reference to GSF "extra" object
    void setGsfExtra( const GsfTrackExtraRef & ref ) { gsfExtra_ = ref; }
    /// reference to "extra" object
    const GsfTrackExtraRef & gsfExtra() const { return gsfExtra_; }

  private:
    /// reference to GSF "extra" extension
    GsfTrackExtraRef gsfExtra_;
  };

}

#endif
