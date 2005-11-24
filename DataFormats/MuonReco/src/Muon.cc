#include "DataFormats/MuonReco/interface/Muon.h"
#include <algorithm>
using namespace std;
using namespace reco;

Muon::Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	    const Parameters & par, const Covariance & cov ) :
  Track( chi2, ndof, found, invalid, lost, par, cov ) {
}

Muon::Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   int q, const Point & v, const Vector & p, 
	    const PosMomError & err ) :
  Track( chi2, ndof, found, invalid, lost, q, v, p, err ) {
}
