#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
// $Id: Muon.h,v 1.5 2005/12/15 20:42:48 llista Exp $
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include"DataFormats/TrackReco/interface/TrackFwd.h"
#include"DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/RecHitFwd.h"
#include"DataFormats/MuonReco/interface/MuonFwd.h"
#include"DataFormats/MuonReco/interface/MuonExtraFwd.h"

namespace reco {
 
  class Muon : public TrackBase {
  public:
    Muon() {}
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	  const Parameters &, const Covariance & );
    Muon( float chi2, unsigned short ndof, int found, int invalid, int lost,
	  int q, const Point & v, const Vector & p, 
	  const PosMomError & err );

    void setExtra( const MuonExtraRef & ref ) { extra_ = ref; }
    const MuonExtraRef & extra() const { return extra_; }

    const Point & outerPosition() const;
    const Vector & outerMomentum() const;
    bool outerOk() const;
    recHit_iterator recHitsBegin() const;
    recHit_iterator recHitsEnd() const;
    size_t recHitsSize() const;
    double outerPx() const;
    double outerPy() const;
    double outerPz() const;
    double outerX() const;
    double outerY() const;
    double outerZ() const;
    double outerP() const;
    double outerPt() const;
    double outerPhi() const;
    double outerEta() const;
    double outerTheta() const;    
    double outerRadius() const;
    const TrackRef & trackerSegment() const;
    const TrackRef & muonSegment() const;

  private:
    MuonExtraRef extra_;
};

}

#endif
