#include "DataFormats/TrackReco/interface/GsfTrack.h"
// #include "DataFormats/TrackReco/interface/GsfTrackExtra.h"
using namespace reco;

GsfTrack::GsfTrack( double chi2, double ndof,
	      const ParameterVector & par, double pt, const CovarianceMatrix & cov ) :
  TrackBase( chi2, ndof, par, pt, cov ) {
}
