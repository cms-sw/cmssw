#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
#include <algorithm>
using namespace reco;

TrackBase::TrackBase( double chi2, double ndof,
		      const ParameterVector & par, double pt, const CovarianceMatrix & cov ) :
  chi2_( chi2 ), ndof_( ndof ), pt_( pt ) {
  std::copy( par.begin(), par.end(), parameters_ );
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = cov( i, j );
}

TrackBase::ParameterVector & TrackBase::fill( ParameterVector & v ) const {
  std::copy( parameters_, parameters_ + dimension, v.begin() );
  return v;
}

TrackBase::CovarianceMatrix & TrackBase::fill( CovarianceMatrix & v ) const {
  return fillCovariance( v, covariance_ );
}
