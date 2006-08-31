#ifndef TrackReco_PerigeeCovariance_h
#define TrackReco_PerigeeCovariance_h
/** \class reco::perigee::Covariance PerigeeCovariance.h DataFormats/TrackReco/interface/PerigeeCovariance.h
 *
 * Model of 5x5 covariance matrix for perigee parameters
 * 
 * \author Thomas Speer, Luca Lista
 *
 */

#include "DataFormats/TrackReco/interface/PerigeeParameters.h"

namespace reco {
  namespace perigee {
    /// perigee parameter covariance matrix (5x5)
    typedef math::Error<5>::type ParameterError;
    /// position-momentum covariance matrix (6x6).
    typedef math::Error<6>::type PosMomError;

    class Covariance {
    public:
      /// default constructor
      Covariance() {} 
      /// constructor from matrix
      Covariance( const ParameterError & e ) : 
	cov_( e ) { }
      /// constructor from double * (15 parameters)
      Covariance( const double * cov ) : cov_() { 
	int k = 0;
	for( int i = 0; i < ParameterError::kRows; ++i )
	  for( int j = i; j < ParameterError::kCols; ++j )
	    cov_( i, j ) = cov[ k++ ];
      }
      /// index type
      typedef unsigned int index;
      /// accessing (i, j)-th parameter, i, j = 0, ..., 4 (read only mode)
      double operator()( index i, index j ) const { return cov_( i, j ); }
      /// accessing (i, j)-th parameter, i, j = 0, ..., 4
      double & operator()( index i, index j ) { return cov_ ( i, j ); }

      /// error on specified element
      double error( index i ) const { return sqrt( cov_( i, i ) ); }

      /// error on the transverse curvature
      double transverseCurvatureError() const { return sqrt( cov_( i_tcurv, i_tcurv ) ); }
      /// error on theta
      double thetaError() const { return sqrt( cov_( i_theta, i_theta ) ); }
      /// error on phi0
      double phi0Error() const { return sqrt( cov_( i_phi0, i_phi0 ) ); }
      /// error on d0
      double d0Error() const { return sqrt( cov_( i_d0, i_d0 ) ); }
      /// error on dx
      double dzError() const { return sqrt( cov_( i_dz, i_dz ) ); }

    private:
      /// 5x5 matrix
      ParameterError cov_;
    };
  }
}


#endif
