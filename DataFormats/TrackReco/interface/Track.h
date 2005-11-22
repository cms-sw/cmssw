#ifndef TrackReco_Track_h
#define TrackReco_Track_h
//
// $Id: Track.h,v 1.12 2005/11/17 18:14:27 llista Exp $
//
// Definition of Track class for RECO
//
// Author: Luca Lista
//
#include <Rtypes.h>
#include "DataFormats/TrackReco/interface/HelixParameters.h"

namespace reco {

  class Track {
  public:
    Track() { }
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   const HelixParameters & );
    unsigned short foundHits() const { return found_; }
    unsigned short lostHits() const { return lost_; }
    unsigned short invalidHits() const { return invalid_; }
    double chi2() const { return chi2_; }
    unsigned short ndof() const { return ndof_; }
    double normalizedChi2() const { return chi2_ / ndof_; }
    const HelixParameters & helix() const { return helix_; }
    int charge() const { return helix_.charge(); }
    double pt() const { return helix_.pt(); }
    double doca() const { return helix_.d0(); }
    typedef HelixParameters::Vector Vector;
    typedef HelixParameters::Point Point;
    Vector momentum() const { return helix_.momentum(); }
    Point poca() const { return helix_.poca(); }
    Error<6> covariance() const { return helix_.posMomError(); }
    unsigned short found() const { return found_; }
    unsigned short lost() const { return lost_; }
    unsigned short invalid() const { return invalid_; }
    double p() const { return momentum().mag(); }
    double px() const { return momentum().x(); }
    double py() const { return momentum().y(); }
    double pz() const { return momentum().z(); }
    double phi() const { return momentum().phi(); }
    double eta() const { return momentum().eta(); }
    double theta() const { return momentum().theta(); }
    double x() const { return poca().x(); }
    double y() const { return poca().y(); }
    double z() const { return poca().z(); }

  private:
    Double32_t chi2_;
    unsigned short ndof_;
    unsigned short found_;
    unsigned short lost_;
    unsigned short invalid_;
    HelixParameters helix_;
  };

}

#endif
