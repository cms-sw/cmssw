#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco;

Track::Track( double chi2, double ndof, const Point & vertex, const Vector & momentum, int charge,
	      const CovarianceMatrix & cov,
	      TrackAlgorithm algo, TrackQuality quality) :
  TrackBase( chi2, ndof, vertex, momentum, charge, cov, algo, quality ) {
}

Track::~Track() {
}

double Track::residualX (int position) const
{
     return extra_->residuals().residualX(position, hitPattern());
}

double Track::residualY (int position) const
{
     return extra_->residuals().residualY(position, hitPattern());
}

