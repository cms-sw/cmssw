#include "DataFormats/TrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
using namespace reco;

GsfTrackExtra::GsfTrackExtra( const Point & outerPosition, const Vector & outerMomentum,
			      const CovarianceMatrix& outerCov,
			      const std::vector<GsfComponent5D>& outerStates, 
			      const double& outerLocalPzSign, unsigned int outerId, bool ok,
			      const Point & innerPosition, const Vector & innerMomentum,
			      const CovarianceMatrix& innerCov,
			      const std::vector<GsfComponent5D>& innerStates, 
			      const double& innerLocalPzSign, unsigned int innerId, bool iok)  :
  TrackExtraBase(),
  outerPosition_( outerPosition ), outerMomentum_( outerMomentum ), 
  outerOk_( ok ), outerDetId_( outerId ),
  innerPosition_( innerPosition ), innerMomentum_( innerMomentum ), 
  innerOk_( iok ), innerDetId_( innerId ),
  outerStates_(outerStates), positiveOuterStatePz_(outerLocalPzSign>0.),
  innerStates_(innerStates), positiveInnerStatePz_(innerLocalPzSign>0.) {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i ) {
    for( index j = 0; j <= i; ++ j ) {
      outerCovariance_[ idx ] = outerCov( i, j );
      innerCovariance_[ idx ] = innerCov( i, j );
      ++idx;
    }
  }
}

GsfTrackExtra::CovarianceMatrix 
GsfTrackExtra::outerStateCovariance() const {
  CovarianceMatrix v; fillCovariance( v, outerCovariance_ ); return v;
}
                                                                                                            
GsfTrackExtra::CovarianceMatrix 
GsfTrackExtra::innerStateCovariance() const {
  CovarianceMatrix v; fillCovariance( v, innerCovariance_ ); return v;
}
                                                                                                            
GsfTrackExtra::CovarianceMatrix & 
GsfTrackExtra::fillOuter( CovarianceMatrix & v ) const {
  return fillCovariance( v, outerCovariance_ );
}
                                                                                                            
GsfTrackExtra::CovarianceMatrix & 
GsfTrackExtra::fillInner( CovarianceMatrix & v ) const {
  return fillCovariance( v, innerCovariance_ );
}
                                                                                                            
std::vector<double> 
GsfTrackExtra::weights (const std::vector<GsfComponent5D>& states) const
{
  std::vector<double> result(states.size());
  std::vector<double>::iterator ir(result.begin());
  for ( std::vector<GsfComponent5D>::const_iterator i=states.begin();
	i!=states.end(); ++i ) {
    *(ir++) = (*i).weight();
  }
  return result;
}

std::vector<GsfTrackExtra::LocalParameterVector> 
GsfTrackExtra::parameters (const std::vector<GsfComponent5D>& states) const
{
  std::vector<LocalParameterVector> result(states.size());
  std::vector<LocalParameterVector>::iterator ir(result.begin());
  for ( std::vector<GsfComponent5D>::const_iterator i=states.begin();
	i!=states.end(); ++i ) {
    *(ir++) = (*i).parameters();
  }
  return result;
}

std::vector<GsfTrackExtra::LocalCovarianceMatrix> 
GsfTrackExtra::covariances (const std::vector<GsfComponent5D>& states) const
{
  std::vector<LocalCovarianceMatrix> result(states.size());
  std::vector<LocalCovarianceMatrix>::iterator ir(result.begin());
  for ( std::vector<GsfComponent5D>::const_iterator i=states.begin();
	i!=states.end(); ++i ) {
    (*i).covariance(*(ir++));
  }
  return result;
}
