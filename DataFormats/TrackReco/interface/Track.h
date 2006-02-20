#ifndef TrackReco_Track_h
#define TrackReco_Track_h
//
// $Id: Track.h,v 1.9 2006/02/17 08:14:58 llista Exp $
//
// Definition of Track class for RECO
//
// Author: Luca Lista
//
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/RecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class Track : public TrackBase {
  public:
    Track() { }
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   const Parameters &, const Covariance & );
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   int q, const Point & v, const Vector & p, 
	   const PosMomError & err );

    void setExtra( const TrackExtraRef & ref ) { extra_ = ref; }
    const TrackExtraRef & extra() const { return extra_; }

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

  private:
    Double32_t chi2_;
    unsigned short ndof_;
    unsigned short found_;
    unsigned short lost_;
    unsigned short invalid_;
    Parameters par_;
    Covariance cov_;
    TrackExtraRef extra_;
  };

}

#endif
