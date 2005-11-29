#ifndef TrackReco_Track_h
#define TrackReco_Track_h
//
// $Id: Track.h,v 1.3 2005/11/24 12:12:08 llista Exp $
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
    typedef helix::Parameters Parameters;
    typedef helix::Covariance Covariance;
    typedef helix::PosMomError PosMomError;
    typedef helix::Point Point;
    typedef helix::Vector Vector;
    Track() { }
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   const Parameters &, const Covariance & );
    Track( float chi2, unsigned short ndof, int found, int invalid, int lost,
	   int q, const Point & v, const Vector & p, 
	   const PosMomError & err );
    double chi2() const { return chi2_; }
    unsigned short ndof() const { return ndof_; }
    double normalizedChi2() const { return chi2_ / ndof_; }
    const Parameters & parameters() const { return par_; }
    double parameter( int i ) const { return par_( i ); }
    const Covariance & covariance() const { return cov_; }
    double covariance( int i, int j ) const { return cov_( i, j ); }
    int charge() const { return par_.charge(); }
    double pt() const { return par_.pt(); }
    double d0() const { return par_.d0(); }
    double phi0() const { return par_.phi0(); }
    double omega() const { return par_.omega(); }
    double dz() const { return par_.dz(); }
    double tanDip() const { return par_.tanDip(); }
    Vector momentum() const { return par_.momentum(); }
    Point vertex() const { return par_.vertex(); }
    PosMomError posMomError() const { return helix::posMomError( par_, cov_ ); }
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
    double x() const { return vertex().x(); }
    double y() const { return vertex().y(); }
    double z() const { return vertex().z(); }

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
