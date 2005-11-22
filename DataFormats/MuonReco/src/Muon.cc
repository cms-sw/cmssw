#include "DataFormats/MuonReco/interface/Muon.h"
#include <algorithm>
using namespace std;
using namespace reco;

Muon::Muon( float chi2, unsigned short ndof, int found, int lost, int invalid,
	    HelixParameters & p ) : 
  Track( chi2, ndof, found, lost, invalid, p ) {
}
