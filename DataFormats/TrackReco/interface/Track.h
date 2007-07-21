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
 * \version $Id: Track.h,v 1.32 2007/01/31 08:51:35 llista Exp $
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
    Track( double chi2, double ndof, const Point & referencePoint,
	   const Vector & momentum, int charge, const CovarianceMatrix & );
    /// return true if the outermost hit is valid
    bool outerOk() const { return extra_->outerOk(); }
    /// return true if the innermost hit is valid
    bool innerOk() const { return extra_->innerOk(); }
    /// position of the innermost hit
    const math::XYZPoint & innerPosition()  const { return extra_->innerPosition(); }

    /// momentum vector at the innermost hit position
    const math::XYZVector & innerMomentum() const { return extra_->innerMomentum(); }
    /// position of the outermost hit
    const math::XYZPoint & outerPosition()  const { return extra_->outerPosition(); }
    /// momentum vector at the outermost hit position
    const math::XYZVector & outerMomentum() const { return extra_->outerMomentum(); }
    /// outermost trajectory state curvilinear errors
    CovarianceMatrix outerStateCovariance() const { return extra_->outerStateCovariance(); }
    /// innermost trajectory state curvilinear errors
    CovarianceMatrix innerStateCovariance() const { return extra_->innerStateCovariance(); }
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillOuter( CovarianceMatrix & v ) const { return extra_->fillOuter( v ); }
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillInner( CovarianceMatrix & v ) const { return extra_->fillInner( v ); }
    /// DetId of the detector on which surface the outermost state is located
    unsigned int outerDetId() const { return extra_->outerDetId(); }
    /// DetId of the detector on which surface the innermost state is located
    unsigned int innerDetId() const { return extra_->innerDetId(); }
    /// first iterator to RecHits
    trackingRecHit_iterator recHitsBegin() const { return extra_->recHitsBegin(); }
    /// last iterator to RecHits
    trackingRecHit_iterator recHitsEnd() const { return extra_->recHitsEnd(); }
    /// get n-th recHit
    TrackingRecHitRef recHit( size_t i ) const { return extra_->recHit( i ); }
    /// number of RecHits
    size_t recHitsSize() const { return extra_->recHitsSize(); }
    /// x coordinate of momentum vector at the outermost hit position
    double outerPx()     const { return extra_->outerPx(); }
    /// y coordinate of momentum vector at the outermost hit position
    double outerPy()     const { return extra_->outerPy(); }
    /// z coordinate of momentum vector at the outermost hit position
    double outerPz()     const { return extra_->outerPz(); }
    /// x coordinate of the outermost hit position
    double outerX()      const { return extra_->outerX(); }
    /// y coordinate of the outermost hit position
    double outerY()      const { return extra_->outerY(); }
    /// z coordinate of the outermost hit position
    double outerZ()      const { return extra_->outerZ(); }
    /// magnitude of momentum vector at the outermost hit position
    double outerP()      const { return extra_->outerP(); }
    /// transverse momentum at the outermost hit position
    double outerPt()     const { return extra_->outerPt(); }
    /// azimuthal angle of the  momentum vector at the outermost hit position
    double outerPhi()    const { return extra_->outerPhi(); }
    /// pseudorapidity of the  momentum vector at the outermost hit position
    double outerEta()    const { return extra_->outerEta(); }
    /// polar angle of the  momentum vector at the outermost hit position
    double outerTheta()  const { return extra_->outerTheta(); }    
    /// polar radius of the outermost hit position
    double outerRadius() const { return extra_->outerRadius(); }
    /// set reference to "extra" object
    void setExtra( const TrackExtraRef & ref ) { extra_ = ref; }
    /// reference to "extra" object
    const TrackExtraRef & extra() const { return extra_; }

    unsigned short found() const { return  numberOfValidHits(); }
   /// number of hits lost
    unsigned short lost() const {return  numberOfLostHits();  }
    /// number of invalid hits
    //    unsigned short invalid() const { return invalid_; }
  private:
    /// reference to "extra" extension
    TrackExtraRef extra_;
  };

}

#endif
