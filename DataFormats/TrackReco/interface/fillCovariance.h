#ifndef TrackReco_fillCovariance_h
#define TrackReco_fillCovariance_h
#include "DataFormats/Math/interface/Error.h"

namespace reco {
  typedef math::Error<5>::type PerigeeCovarianceMatrix;
  PerigeeCovarianceMatrix & fillCovariance( PerigeeCovarianceMatrix & v, const float * data );
}
  
#endif
