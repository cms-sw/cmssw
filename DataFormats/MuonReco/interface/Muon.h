#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
// $Id: Muon.h,v 1.1 2005/11/22 14:05:45 llista Exp $
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/Error.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Vector3D.h>

namespace reco {

  class Muon : public Track {
  public:
    Muon() {}
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   const Parameters &, const Covariance & );
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   int q, const Point & v, const Vector & p, 
	   const PosMomError & err );
  };

}

#endif
