#include "DataFormats/TrackReco/interface/GsfTrack.h"
using namespace reco;

GsfTrack::GsfTrack( double chi2, double ndof, const Point & vertex, const Vector & momentum, int charge,
		    const CovarianceMatrix & cov ) :
  TrackBase( chi2, ndof, vertex, momentum, charge, cov ) {
}
