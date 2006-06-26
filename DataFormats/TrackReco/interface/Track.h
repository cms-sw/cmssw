#ifndef TrackReco_Track_h
#define TrackReco_Track_h
/** \class reco::Track Track.h DataFormats/TrackReco/interface/Track.h
 *
 * Reconstructed Track. It is ment to be stored
 * in the AOD, with a reference to an extension
 * object stored in the RECO
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Track.h,v 1.20 2006/06/14 06:33:20 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class Track : public TrackBase {
  public:
    /// default constructor
    Track() { }
    /// constructor from fit parameters and error matrix
    /// notice that the reference point must be 
    /// the point of closest approch to the beamline.    
    Track( double chi2, double ndof,
	   const ParameterVector & par, double pt, const CovarianceMatrix & cov );
    /// return true if the outermost point is valid
    bool outerOk() const { return extra_->outerOk(); }
    /// position of the outermost point
    const math::XYZPoint & outerPosition()  const { return extra_->outerPosition(); }
    /// momentum vector at the outermost point
    const math::XYZVector & outerMomentum() const { return extra_->outerMomentum(); }
    /// first iterator to RecHits
    trackingRecHit_iterator recHitsBegin() const { return extra_->recHitsBegin(); }
    /// last iterator to RecHits
    trackingRecHit_iterator recHitsEnd()   const { return extra_->recHitsEnd(); }
    /// get n-th recHit
    TrackingRecHitRef recHit( size_t i ) const { return extra_->recHit( i ); }
    /// number of RecHits
    size_t recHitsSize() const { return extra_->recHitsSize(); }
    /// x coordinate of momentum vector at the outermost point
    double outerPx()     const { return extra_->outerPx(); }
    /// y coordinate of momentum vector at the outermost point
    double outerPy()     const { return extra_->outerPy(); }
    /// z coordinate of momentum vector at the outermost point
    double outerPz()     const { return extra_->outerPz(); }
    /// x coordinate of the outermost point
    double outerX()      const { return extra_->outerX(); }
    /// y coordinate of the outermost point
    double outerY()      const { return extra_->outerY(); }
    /// z coordinate of the outermost point
    double outerZ()      const { return extra_->outerZ(); }
    /// magnitude of momentum vector at the outermost point
    double outerP()      const { return extra_->outerP(); }
    /// transverse momentum at the outermost point
    double outerPt()     const { return extra_->outerPt(); }
    /// azimuthal angle of the  momentum vector at the outermost point
    double outerPhi()    const { return extra_->outerPhi(); }
    /// pseudorapidity of the  momentum vector at the outermost point
    double outerEta()    const { return extra_->outerEta(); }
    /// polar angle of the  momentum vector at the outermost point
    double outerTheta()  const { return extra_->outerTheta(); }    
    /// polar radius of the outermost point
    double outerRadius() const { return extra_->outerRadius(); }
    /// set reference to "extra" object
    void setExtra( const TrackExtraRef & ref ) { extra_ = ref; }
    /// reference to "extra" object
    const TrackExtraRef & extra() const { return extra_; }

  private:
    /// reference to "extra" extension
    TrackExtraRef extra_;
  };

}

#endif
