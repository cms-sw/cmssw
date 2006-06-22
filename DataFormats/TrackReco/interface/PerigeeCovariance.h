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
#include "DataFormats/Math/interface/Error.h"
#include <algorithm>

namespace reco {
  namespace perigee {
    /// 5 parameter covariance matrix
    typedef math::Error<dimension>::type ParameterError;
    /// position-momentum covariance matrix (6x6).
    typedef math::Error<6>::type PosMomError;
    class Covariance {
    public:
      /// matrix size
      enum { size = ParameterError::kSize };
      /// default constructor
      Covariance() : cov_( size ) { }
      /// constructor from matrix
      Covariance( const ParameterError & e );
      /// constructor from double * (15 parameters)
      Covariance( const double * cov ) : cov_( size ) { 
	std::copy( cov, cov + ParameterError::kSize, cov_.begin() );
      }
      /// index type
      typedef unsigned int index;
      /// accessing (i, j)-th parameter, i, j = 0, ..., 4 (read only mode)
      double operator()( index i, index j ) const { return cov_[ idx( i, j ) ]; }
      /// accessing (i, j)-th parameter, i, j = 0, ..., 4
      double & operator()( index i, index j ) { return cov_[ idx ( i, j ) ]; }
      /// error on specified element
      double error( index i ) const { return sqrt( cov_[ idx( i, i ) ] ); }
      /// error on the transverse curvature
      double transverseCurvatureError() const { return sqrt( cov_[ idx( i_tcurv, i_tcurv ) ] ); }
      /// error on theta
      double thetaError() const { return sqrt( cov_[ idx( i_theta, i_theta ) ] ); }
      /// error on phi0
      double phi0Error() const { return sqrt( cov_[ idx ( i_phi0, i_phi0 ) ] ); }
      /// error on d0
      double d0Error() const { return sqrt( cov_[ idx( i_d0, i_d0 ) ] ); }
      /// error on dx
      double dzError() const { return sqrt( cov_[ idx( i_dz, i_dz ) ] ); }
      /// return SMatrix
      ParameterError matrix() const { ParameterError m; fill( m ); return m; }
      /// fill SMatrix
      void fill( ParameterError & v ) const;

    private:
      /// 5x5 matrix as array
      std::vector<Double32_t> cov_;
      /// position index
      index idx( index i, index j ) const {
	int a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
	return a * dimension + b - a * ( a + 1 ) / 2;
      }
    };
  }
}

#endif
