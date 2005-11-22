#include "DataFormats/TrackReco/interface/TrackExtra.h"
using namespace reco;

TrackExtra::TrackExtra( double x, double y, double z, 
			double px, double py, double pz, 
			bool ok ) :
  outerPosition_(), outerMomentum_(), outerOk_( ok ) {
  outerPosition_.get<0>() = x;
  outerPosition_.get<1>() = y;
  outerPosition_.get<2>() = z;
  outerMomentum_.get<0>() = px;
  outerMomentum_.get<1>() = py;
  outerMomentum_.get<2>() = pz;
}


