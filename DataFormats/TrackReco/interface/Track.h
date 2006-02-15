#ifndef TrackReco_Track_h
#define TrackReco_Track_h
//
// $Id: Track.h,v 1.7 2006/01/20 10:03:34 llista Exp $
//
// Definition of Track class for RECO
//
// Author: Luca Lista
//
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/TrackReco/interface/HelixParameters.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {

  class Track {
  public:
    typedef helix::Parameters Parameters;
    typedef helix::Covariance Covariance;
    typedef math::Error<6> PosMomError;
    typedef math::XYZVector Vector;
    typedef math::XYZPoint Point;

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
    double p() const { return momentum().R(); }
    double px() const { return momentum().X(); }
    double py() const { return momentum().Y(); }
    double pz() const { return momentum().Z(); }
    double phi() const { return momentum().Phi(); }
    double eta() const { return momentum().Eta(); }
    double theta() const { return momentum().Theta(); }
    double x() const { return vertex().X(); }
    double y() const { return vertex().Y(); }
    double z() const { return vertex().Z(); }

    void setExtra( const TrackExtraRef & ref ) { extra_ = ref; }
    const TrackExtraRef & extra() const { return extra_; }
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
