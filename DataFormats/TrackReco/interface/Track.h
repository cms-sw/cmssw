#ifndef TrackReco_Track_h
#define TrackReco_Track_h
//
// $Id: Track.h,v 1.10 2006/02/20 14:42:00 llista Exp $
//
// Definition of Track class for RECO
//
// Author: Luca Lista
//
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtension.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class Track : public TrackBase, TrackExtension<TrackExtraRef> {
  public:
    Track() { }
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   const Parameters &, const Covariance & );
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   int q, const Point & v, const Vector & p, 
	   const PosMomError & err );

  private:
    Double32_t chi2_;
    unsigned short ndof_;
    unsigned short found_;
    unsigned short lost_;
    unsigned short invalid_;
    Parameters par_;
    Covariance cov_;
  };

}

#endif
