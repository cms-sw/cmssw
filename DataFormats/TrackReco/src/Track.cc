#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
using namespace reco;

Track::Track( double chi2, double ndof,
	      const ParameterVector & par, double pt, const CovarianceMatrix & cov ) :
  TrackBase( chi2, ndof, par, pt, cov ) {
}
