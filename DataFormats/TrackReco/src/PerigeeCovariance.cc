#include "DataFormats/TrackReco/interface/PerigeeCovariance.h"
// $Id$
using namespace reco::perigee;

void Covariance::fill( ParameterError & v ) const {
  index idx = 0;
  for( index i = 0; i < ParameterError::kSize; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      v( i, j ) = cov_[ idx ++ ];
}
