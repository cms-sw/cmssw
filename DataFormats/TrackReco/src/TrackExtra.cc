#include "DataFormats/TrackReco/interface/TrackExtra.h"
using namespace reco;

TrackExtra::TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ) : 
  TrackExtraBase(),
  outerPosition_( outerPosition ), outerMomentum_( outerMomentum ), outerOk_( ok ), innerOk_(false) {
}

TrackExtra::TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ,
			const Point & innerPosition, const Vector & innerMomentum, bool iok ):
   TrackExtraBase(),
   outerPosition_( outerPosition ), outerMomentum_( outerMomentum ), outerOk_( ok ), innerPosition_( innerPosition ), innerMomentum_( innerMomentum ), innerOk_(iok){
}

