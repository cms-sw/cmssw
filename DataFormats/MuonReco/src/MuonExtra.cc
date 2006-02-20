#include "DataFormats/MuonReco/interface/MuonExtra.h"
using namespace reco;

MuonExtra::MuonExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ) :
  TrackExtra( outerPosition, outerMomentum, ok ) {
}

