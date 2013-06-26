#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
using namespace reco;

GsfTrack::GsfTrack ()
{
  chargeMode_ = 0;
  momentumMode_ = Vector(0.,0.,0.);
  typedef unsigned int index;
  index idx = 0;
  for( index i = 0; i < dimensionMode; ++ i )
    for( index j = 0; j <= i; ++ j )
      covarianceMode_[ idx ++ ] = 0.;  
}

GsfTrack::GsfTrack( double chi2, double ndof, const Point & vertex, const Vector & momentum, int charge,
		    const CovarianceMatrix & cov ) :
  Track( chi2, ndof, vertex, momentum, charge, cov ),
  chargeMode_(charge), momentumMode_(momentum) {
  typedef unsigned int index;
  index idx = 0;
  for( index i = 0; i < dimensionMode; ++ i )
    for( index j = 0; j <= i; ++ j )
      covarianceMode_[ idx ++ ] = cov(i,j);
}

void 
GsfTrack::setMode (int chargeMode, const Vector& momentumMode,
		   const CovarianceMatrixMode& covarianceMode)
{
  chargeMode_ = chargeMode;
  momentumMode_ = momentumMode;
  typedef unsigned int index;
  index idx = 0;
  for( index i = 0; i < dimensionMode; ++ i )
    for( index j = 0; j <= i; ++ j )
      covarianceMode_[ idx ++ ] = covarianceMode(i,j);
}

GsfTrack::CovarianceMatrixMode&
GsfTrack::fill (CovarianceMatrixMode& v) const
{
  typedef unsigned int index;
  index idx = 0;
  for( index i = 0; i < dimensionMode; ++ i ) 
    for( index j = 0; j <= i; ++ j )
      v( i, j ) = covarianceMode_[ idx ++ ];
  return v;
}

