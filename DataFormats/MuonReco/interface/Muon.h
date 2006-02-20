#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
// $Id: Muon.h,v 1.6 2006/02/20 14:41:58 llista Exp $
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"
#include"DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include"DataFormats/TrackReco/interface/TrackExtension.h"
#include"DataFormats/MuonReco/interface/MuonFwd.h"
#include"DataFormats/MuonReco/interface/MuonExtraFwd.h"

namespace reco {
 
  class Muon : public TrackBase, public TrackExtension<MuonExtraRef> {
  public:
    Muon() {}
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	  const Parameters &, const Covariance & );
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	  int q, const Point & v, const Vector & p, 
	  const PosMomError & err );

    const TrackRef & trackerSegment() const;
    const TrackRef & muonSegment() const;

  private:
    MuonExtraRef extra_;
};

}

#endif
