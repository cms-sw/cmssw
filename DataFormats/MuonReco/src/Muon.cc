#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonExtra.h"
#include <algorithm>
using namespace std;
using namespace reco;

Muon::Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	    const Parameters & p, const Covariance & c ) :
  TrackBase( chi2, ndof, found, invalid, lost, p, c ) {
}

Muon::Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	    int q, const Point & v, const Vector & p, 
	    const PosMomError & err ) :
  TrackBase( chi2, ndof, found, invalid, lost, q, v, p, err ) {
}


