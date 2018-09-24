#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco;

Track::Track(double chi2, double ndof, const Point &vertex, const Vector &momentum,
             int charge, const CovarianceMatrix &cov, TrackAlgorithm algo,
             TrackQuality quality, float t0, float beta, 
	     float covt0t0, float covbetabeta) :
             TrackBase(chi2, ndof, vertex, momentum, charge, cov, algo, quality,
		       0,0, // nloops and stop reason
		       t0,beta,covt0t0,covbetabeta)
{
    ;
}

Track::~Track()
{

}

