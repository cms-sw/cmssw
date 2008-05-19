#include "Rtypes.h" 
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/fillCovariance.h"
#include <algorithm>
using namespace reco;

TrackBase::TrackBase() :
  chi2_(0), ndof_(0), vertex_(0,0,0), momentum_(0,0,0), charge_(0), algorithm_(undefAlgorithm), quality_(0) {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i )
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ]=0;
}

TrackBase::TrackBase( double chi2, double ndof, const Point & vertex, const Vector & momentum, int charge,
		      const CovarianceMatrix & cov,
		      TrackAlgorithm algorithm , TrackQuality quality) :
  chi2_( chi2 ), ndof_( ndof ), vertex_( vertex ), momentum_( momentum ), charge_( charge ), algorithm_(algorithm), quality_(0) {
  index idx = 0;
  for( index i = 0; i < dimension; ++ i )
    for( index j = 0; j <= i; ++ j )
      covariance_[ idx ++ ] = cov( i, j );
  setQuality(quality);
}

TrackBase::~TrackBase() {
}

TrackBase::CovarianceMatrix & TrackBase::fill( CovarianceMatrix & v ) const {
  return fillCovariance( v, covariance_ );
}

TrackBase::TrackQuality TrackBase::qualityByName(const std::string &name){
  if (name ==  "loose")    return loose;
  else if (name ==  "tight")    return tight;
  else if (name ==  "highPurity")    return highPurity;
  else if (name ==  "confirmed")    return confirmed;
  else if (name ==  "goodIterative")    return goodIterative;
  else if (name ==  "undefQuality")  return undefQuality;
  else return undefQuality; // better this or throw() ?
}


