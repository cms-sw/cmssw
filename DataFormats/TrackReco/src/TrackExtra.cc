#include "DataFormats/TrackReco/interface/TrackExtra.h"
using namespace reco;

TrackExtra::TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok,
			const CovarianceMatrix& outerCov, unsigned int outerId) : 
  TrackExtraBase(),
  outerPosition_( outerPosition ), outerMomentum_( outerMomentum ), outerOk_( ok ), 
  innerOk_(false) 
{
  index idx = 0;
  for( index i = 0; i < 5; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      outerCovariance_[ idx ++ ] = outerCov( i, j );
}

TrackExtra::TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ,
			const Point & innerPosition, const Vector & innerMomentum, bool iok,
			const CovarianceMatrix& outerCov, unsigned int outerId,
			const CovarianceMatrix& innerCov, unsigned int innerId):
  TrackExtraBase(),
  outerPosition_( outerPosition ), outerMomentum_( outerMomentum ), outerOk_( ok ), 
  outerDetId_(outerId),
  innerPosition_( innerPosition ), innerMomentum_( innerMomentum ), innerOk_(iok),
  innerDetId_(innerId)
{
  index idx = 0;
  for( index i = 0; i < 5; ++ i ) {
    for( index j = 0; j <= i; ++ j ) {
      outerCovariance_[ idx] = outerCov( i, j );
      innerCovariance_[ idx] = innerCov( i, j );
      ++idx;
    }
  }
}

TrackExtra::CovarianceMatrix TrackExtra::covariance( const Double32_t * data) const
{
  CovarianceMatrix v;
  index idx = 0;
  for( index i = 0; i < 5; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      v( i, j ) = data[ idx ++ ];
  return v;
}

