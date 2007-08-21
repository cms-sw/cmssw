#include "Rtypes.h" 
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
#include <algorithm>
using namespace reco;

TrackBase::TrackBase( double chi2, double ndof, const Point & vertex, const Vector & momentum, int charge,
		      const CovarianceMatrix & cov ) :
  chi2_( chi2 ), ndof_( ndof ), vertex_( vertex ), momentum_( momentum ), charge_( charge ) {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = cov( i, j );
}

TrackBase::CovarianceMatrix & TrackBase::fill( CovarianceMatrix & v ) const {
  return fillCovariance( v, covariance_ );
}
