#include "DataFormats/TrackReco/interface/TrackBase.h"
#include <algorithm>
using namespace reco;

TrackBase::TrackBase( double chi2, double ndof,
		      const ParameterVector & par, double pt, const CovarianceMatrix & cov ) :
  chi2_( chi2 ), ndof_( ndof ), 
  parameters_( dimension ), pt_( pt ), covariance_( covarianceSize ) {
  std::copy( par.begin(), par.end(), parameters_.begin() );
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = cov( i, j );
}

void TrackBase::fill( ParameterVector & v ) const {
  std::copy( parameters_.begin(), parameters_.end(), v.begin() );
}

void TrackBase::fill( CovarianceMatrix & v ) const {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      v( i, j ) = covariance_[ idx ++ ];
}
