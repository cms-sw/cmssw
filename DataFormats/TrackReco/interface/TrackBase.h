#ifndef TrackReco_TrackBase_h
#define TrackReco_TrackBase_h
/** \class reco::TrackBase TrackBase.h DataFormats/TrackReco/interface/TrackBase.h
 *
 * Common base class to all track types, including Muon.
 * It provides fit parameters, chi-square and n.d.o.f,
 * and summary information of the hit patterm
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TrackBase.h,v 1.10 2006/05/02 13:47:37 llista Exp $
 *
 */
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/PerigeeParameters.h"
#include "DataFormats/TrackReco/interface/PerigeeCovariance.h"

namespace reco {

  class TrackBase {
  public:
    /// perigee fit parameters
    typedef perigee::Parameters Parameters;
    /// perigee parameters covariance matrix (5x5)
    typedef perigee::Covariance Covariance;
    /// position-momentum covariance matrix (6x6).
    /// This type will be replaced by a MathCore symmetric
    /// matrix, as soon as available
    typedef math::Error<6>::type PosMomError;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;

    /// default constructor
    TrackBase() { }
    /// constructor from fit parameters and error matrix
    TrackBase( double chi2, double ndof, 
	       int found, int invalid, int lost,
	       const Parameters &, const Covariance & );
    /// chi-squared of the fit
    double chi2() const { return chi2_; }
    /// number of degrees of freedom of the fit
    double ndof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return chi2_ / ndof_; }
    /// perigee fit parameters
    const Parameters & parameters() const { return par_; }
    /// i-th fit parameter ( i = 0, ... 4 )
    double parameter( int i ) const { return par_( i ); }
    /// covariance matrix of the fit parameters
    const Covariance & covariance() const { return cov_; }
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 4 )
    double covariance( int i, int j ) const { return cov_( i, j ); }

    /// track electric charge
    int charge() const { return par_.charge(); }

    /// track transverse momentum
    double pt() const { return par_.pt(); }

    /// The signed transverse curvature
    double transverseCurvature() const { return par_.transverseCurvature(); }
    /// track azimutal angle of point of closest approach to beamline
    double phi0() const { return par_.phi0(); }
    /// polar angle  
    double theta() const { return par_.theta(); }
    /// track impact parameter (distance of closest approach to beamline)
    double d0() const { return par_.d0(); }
    /// z coordniate of point of closest approach to beamline
    double dz() const { return par_.dz(); }

    /// error on specified element
    double error( int i ) const { return cov_.error( i ); }

    /// error on signed transverse curvature
    double transverseCurvatureError() const { return cov_.transverseCurvatureError(); }
    /// error on theta
    double thetaError() const { return cov_.thetaError(); }
    /// error on phi0
    double phi0Error() const { return cov_.phi0Error(); }
    /// error on d0
    double d0Error() const { return cov_.d0Error(); }
    /// error on dx
    double dzError() const { return cov_.dzError(); }

    /// track momentum vector
    Vector momentum() const { return par_.momentum(); }
    /// position of point of closest approach to the beamline
    Point vertex() const { return par_.vertex(); }
//     /// position-momentum error matrix ( 6x6, degenerate )
//     PosMomError posMomError() const;
    /// number of hits found 
    unsigned short found() const { return found_; }
    /// number of hits lost
    unsigned short lost() const { return lost_; }
    /// number of invalid hits
    unsigned short invalid() const { return invalid_; }
    /// momentum vector magnitude
    double p() const { return momentum().R(); }
    /// x coordinate of momentum vector
    double px() const { return momentum().X(); }
    /// y coordinate of momentum vector
    double py() const { return momentum().Y(); }
    /// z coordinate of momentum vector
    double pz() const { return momentum().Z(); }
    /// azimuthal angle of momentum vector
    double phi() const { return momentum().Phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return momentum().Eta(); }
    /// x coordinate of point of closest approach to the beamline
    double x() const { return vertex().X(); }
    /// y coordinate of point of closest approach to the beamline
    double y() const { return vertex().Y(); }
    /// z coordinate of point of closest approach to the beamline
    double z() const { return vertex().Z(); }
    /// x-y distance of closest approach to the beamline
    double rho() const { return d0(); }

  private:
    /// chi-squared
    Double32_t chi2_;
    /// number of degrees of freedom
    Double32_t ndof_;
    /// number of hits found
    unsigned short found_;
    /// number of hits lost
    unsigned short lost_;
    /// number of invalid hits
    unsigned short invalid_;
    /// perigee 5 parameters
    Parameters par_;
    /// perigee 5x5 covariance matrix
    Covariance cov_;
  };

//   inline TrackBase::PosMomError TrackBase::posMomError() const { 
//     return perigee::posMomError( par_, cov_ ); 
//   }

}

#endif
