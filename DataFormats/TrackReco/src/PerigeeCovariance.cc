#include "DataFormats/TrackReco/interface/PerigeeCovariance.h"
// $Id: PerigeeCovariance.cc,v 1.1 2006/06/22 18:15:53 llista Exp $
using namespace reco::perigee;

Covariance::Covariance( const ParameterError & v ) : 
  cov_( dimension ) { 
  index idx = 0;
  for( index i = 0; i < ParameterError::kSize; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      cov_[ idx ++ ] = v( i, j );
}

void Covariance::fill( ParameterError & v ) const {
  index idx = 0;
  for( index i = 0; i < ParameterError::kSize; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      v( i, j ) = cov_[ idx ++ ];
}
