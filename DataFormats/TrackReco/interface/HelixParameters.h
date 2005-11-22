#ifndef TrackReco_HelixParameters_h
#define TrackReco_HelixParameters_h
/*----------------------------
 $Id: HelixParameters.h,v 1.4 2005/11/17 08:56:11 llista Exp $
 Helix Track Parametrization

 Author: Luca Lista, INFN
------------------------------*/
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Point3D.h>
#include "DataFormats/TrackReco/interface/Error.h"
#include "DataFormats/TrackReco/interface/Vector.h"

namespace reco {

  class HelixParameters {
  public:
    enum index { i_d0 = 0, i_phi0, i_omega, i_dz, i_tanDip }; 
    typedef Vector<5> Parameters;
    typedef reco::Error<5> Error;
    typedef HepGeom::Vector3D<double> Vector;
    typedef HepGeom::Point3D<double> Point;
    typedef reco::Error<6> PosMomError;
    HelixParameters();
    HelixParameters( const Parameters & v, const Error & e ) :
      par_( v ), error_( e ) { }
    HelixParameters( int q, const Point &, const Vector &, const PosMomError & cov ); 
    double d0() const { return par_.get<i_d0>(); }
    double phi0() const { return par_.get<i_phi0>(); }
    double omega() const { return par_.get<i_omega>(); }
    double dz() const { return par_.get<i_dz>(); }
    double tanDip() const { return par_.get<i_tanDip>(); }
    const Parameters & parameters() const { return par_; }
    const Error & covariance() const { return error_; }
    int charge() const;
    double pt() const;
    Vector momentum() const;
    // poca == Point of Closest Approach
    Point poca() const;
    PosMomError posMomError() const;

  private:
    Parameters par_;
    Error error_;
  };
  
  inline HelixParameters::HelixParameters() { }

  inline int HelixParameters::charge() const {
    return omega() > 0 ? +1 : -1;
  }

  inline double HelixParameters::pt() const {
    return 1./ fabs( omega() );
  }

  inline HelixParameters::Vector HelixParameters::momentum() const {
    double p_t = pt();
    return Vector( - p_t * sin( phi0() ), p_t * cos( phi0() ), p_t * tanDip() );
  }

  inline HelixParameters::Point HelixParameters::poca() const {
    return Point( d0() * cos( phi0() ), d0() * sin( phi0() ), dz() );
  }

}


#endif
