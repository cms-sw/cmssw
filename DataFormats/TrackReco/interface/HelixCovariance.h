#ifndef TrackReco_HelixCovariance_h
#define TrackReco_HelixCovariance_h
/** \class reco::helix::Covariance HelixCovariance.h DataFormats/TrackReco/interface/HelixCovariance.h
 *
 * Model of 5x5 covariance matrix for helix parameters
 * according to how described in the following document:
 *
 *   http://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/main.html
 * 
 * \author Luca Lista, INFN
 *
 * \version $Id: HelixParameters.h,v 1.10 2006/04/03 11:59:29 llista Exp $
 *
 */

#include "DataFormats/TrackReco/interface/HelixParameters.h"

namespace reco {
  namespace helix {
    /// helix parameter covariance matrix (5x5)
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
      /// error on d0
      double d0Error() const { return sqrt( cov_( i_d0, i_d0 ) ); }
      /// error on phi0
      double phi0Error() const { return sqrt( cov_( i_phi0, i_phi0 ) ); }
      /// error on omega
      double omegaError() const { return sqrt( cov_( i_omega, i_omega ) ); }
      /// error on dx
      double dzError() const { return sqrt( cov_( i_dz, i_dz ) ); }
      /// error on tanDip
      double tanDipError() const { return sqrt( cov_( i_tanDip, i_tanDip ) ); }

    private:
      /// 5x5 matrix
      ParameterError cov_;
    };
  }
}


#endif
