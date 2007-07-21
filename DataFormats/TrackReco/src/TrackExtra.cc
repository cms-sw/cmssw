#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
using namespace reco;

TrackExtra::TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ,
			const Point & innerPosition, const Vector & innerMomentum, bool iok,
			const CovarianceMatrix& outerCov, unsigned int outerId,
			const CovarianceMatrix& innerCov, unsigned int innerId):
  TrackExtraBase(),
  outerPosition_( outerPosition ), outerMomentum_( outerMomentum ), outerOk_( ok ), 
  outerDetId_( outerId ),
  innerPosition_( innerPosition ), innerMomentum_( innerMomentum ), innerOk_( iok ),
  innerDetId_( innerId ) {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) {
    for( index j = 0; j <= i; ++ j ) {
      outerCovariance_[ idx ] = outerCov( i, j );
      innerCovariance_[ idx ] = innerCov( i, j );
      ++idx;
    }
  }
}

TrackExtra::CovarianceMatrix TrackExtra::outerStateCovariance() const {
  CovarianceMatrix v; fillCovariance( v, outerCovariance_ ); return v;
}

TrackExtra::CovarianceMatrix TrackExtra::innerStateCovariance() const {
  CovarianceMatrix v; fillCovariance( v, innerCovariance_ ); return v;
}

TrackExtra::CovarianceMatrix & TrackExtra::fillOuter( CovarianceMatrix & v ) const {
  return fillCovariance( v, outerCovariance_ );
}

TrackExtra::CovarianceMatrix & TrackExtra::fillInner( CovarianceMatrix & v ) const {
  return fillCovariance( v, innerCovariance_ );
}

