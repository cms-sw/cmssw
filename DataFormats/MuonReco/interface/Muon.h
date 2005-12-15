#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
// $Id: Muon.h,v 1.4 2005/12/11 17:58:15 llista Exp $
#include "DataFormats/TrackReco/interface/Track.h"
#include"DataFormats/MuonReco/interface/MuonFwd.h"

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
