#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
using namespace reco;

GsfTrack::GsfTrack( double chi2, double ndof, const Point & vertex, const Vector & momentum, int charge,
		    const CovarianceMatrix & cov ) :
  Track( chi2, ndof, vertex, momentum, charge, cov ) {
}
