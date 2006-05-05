#ifndef TrackReco_HelixParameters_h
#define TrackReco_HelixParameters_h
/** \class reco::helix::Parameters HelixParameters.h DataFormats/TrackReco/interface/HelixParameters.h
 *
 * Model of 5 helix parameters for Track fit 
 * according to how described in the following document:
 *
 *   http://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/main.html
 * 
 * \author Luca Lista, INFN
 *
 * \version $Id: HelixParameters.h,v 1.11 2006/04/19 13:35:05 llista Exp $
 *
 */

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Vector.h"
#include <cmath>

namespace reco {
  namespace helix {
    /// enumerator provided indices to the five parameters
    enum index { i_d0 = 0, i_phi0, i_omega, i_dz, i_tanDip }; 
    /// parameter vector
    typedef math::Vector<5>::type ParameterVector;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
 
    class Parameters {
    public:
      /// default constructor
      Parameters() { }
      /// constructor from five double parameters
      Parameters( double p1, double p2, double p3, double p4, double p5 ) : 
	par_( p1, p2, p3, p4, p5 ) { }
      /// constructor from vector
      Parameters( const ParameterVector p ) : 
	par_( p ) { }
      /// constructor from double *
      Parameters( const double * p ) : 
	par_( p[ 0 ], p[ 1 ], p[ 2 ], p[ 3 ], p[ 4 ] ) { }
       /// index type
      typedef unsigned int index;
      /// accessing i-th parameter, i = 0, ..., 4 (read-only mode)
      double operator()( index i ) const { return par_( i ); }
      /// accessing i-th parameter, i = 0, ..., 4
      double & operator()( index i ) { return par_( i ); }
      /// track impact parameter (distance of closest approach to beamline) (read-only mode)
      double d0() const { return par_[ i_d0 ]; }
      /// track azimuthal angle of point of closest approach to beamline (read-only mode)
      double phi0() const { return par_[ i_phi0 ]; }
      /// e / pt (electric charge divided by transverse momentum) (read-only mode)     
      double omega() const { return par_[ i_omega ]; }
      /// z coordniate of point of closest approach to beamline (read-only mode)
      double dz() const { return par_[ i_dz ]; }
      /// tangent of the dip angle ( tanDip = pz / pt ) (read-only mode)
      double tanDip() const { return par_[ i_tanDip ]; }
      /// track impact parameter (distance of closest approach to beamline)
      double & d0() { return par_[ i_d0 ]; }
      /// track azimuthal angle of point of closest approach to beamline
      double & phi0() { return par_[ i_phi0 ]; }
      /// e / pt (electric charge divided by transverse momentum)      
      double & omega() { return par_[ i_omega ]; }
      /// z coordniate of point of closest approach to beamline
      double & dz() { return par_[ i_dz ]; }
      /// tangent of the dip angle ( tanDip = pz / pt )
      double & tanDip() { return par_[ i_tanDip ]; }
      /// electric charge
      int charge() const;
      /// transverse momentum
      double pt() const;
      /// momentum vector
      Vector momentum() const;
      /// position of point of closest approach to the beamline
      Point vertex() const;
      
    private:
      /// five parameters
      ParameterVector par_;
    };

    inline int Parameters::charge() const {
      return omega() > 0 ? +1 : -1;
    }
    
    inline double Parameters::pt() const {
      return 1./ fabs( omega() );
    }
    
    inline Vector Parameters::momentum() const {
      double p_t = pt();
      return Vector( pt() * cos( phi0() ), pt() * sin( phi0() ), p_t * tanDip() );
    }
    
    inline Point Parameters::vertex() const {
      return Point( d0() * sin( phi0() ), - d0() * cos( phi0() ), dz() );
    }

  }
}

#endif
