#ifndef TrackReco_PerigeeParameters_h
#define TrackReco_PerigeeParameters_h
/** \class reco::perigee::Parameters PerigeeParameters.h DataFormats/TrackReco/interface/PerigeeParameters.h
 *
 * Model of 5 perigee parameters for Track fit <BR>
 * <B> (q/R, theta, phi_0, d_0, z_0) </B><BR>
 * defined as:  <BR>
 *   <DT> q/R = charge unit divided by radius of curvature in transverse plane </DT> 
 *   <DT> theta = polar angle at pca. to the beam line </DT>
 *   <DT> phi_0 = azimuth angle at pca. to the beam line </DT>
 *   <DT> d_0 = signed transverse dca. to the beam line (positive if the beam is outside the circle) </DT>
 *   <DT> z_0 = z-coordinate of pca. to the beam line </DT>
 *
 * according to how described in the following documents: <BR>
 * <a href="http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=NOTE&year=2006&files=NOTE2006_001.pdf">A. Strandlie, W. Wittek, "Propagation of Covariance Matrices...", CMS Note 2006/001</a> <BR>
 * P. Billoir, S. Qian, "Fast vertex fitting...", NIM A311 (1992) 139. <BR>
 * <a href="http://cmsdoc.cern.ch/cms/Physics/btau/management/activities/reconstruction/vertex/tutorial041112.d/node5.html#SECTION00053000000000000000">Track parametrization in vertex fitting (Vertex reconstruction tutorial)</a> <BR>
 * 
 * \author Thomas Speer,  Luca Lista, Pascal Vanlaer
 *
 */

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace reco {
  namespace perigee {
    /// enumerator provided indices to the five parameters
    enum index { i_tcurv = 0 , i_theta, i_phi0, i_d0, i_dz}; 
    /// parameter vector internal storage
    typedef std::vector<Double32_t> InnerParameterVector;
    /// parameter dimension
    enum { dimension = 5 };
    /// parameter vector<
    typedef math::Vector<dimension>::type ParameterVector;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
 
    class Parameters {
    public:
      /// default constructor
      Parameters() : par_( dimension ) { }
      /// constructor from five double parameters
      //      Parameters( double p1, double p2, double p3, double p4, double p5 , double pt) : 
	//	par_( p1, p2, p3, p4, p5 ), pt_(pt) { }
      Parameters( double p0, double p1, double p2, double p3, double p4, double pt ) : 
	par_( dimension ), pt_( pt ) {
	par_[ 0 ] = p0;	par_[ 1 ] = p1;	par_[ 2 ] = p2;	par_[ 3 ] = p3;	par_[ 4 ] = p4;
      }
      /// constructor from vector
      Parameters( const InnerParameterVector & p , double pt ) : 
	par_( p ), pt_( pt ) { }
      /// constructor from double *
      Parameters( const double * p, double pt ) : par_( dimension ), pt_( pt ) {
	std::copy( p, p + dimension, par_.begin() );
      }
      /// index type
      typedef unsigned int index;
      /// accessing i-th parameter, i = 0, ..., 4 (read-only mode)
      //      double operator()( index i ) const { return par_( i ); }
      double operator()( index i ) const { return par_[i]; }
      /// accessing i-th parameter, i = 0, ..., 4
      //      double & operator()( index i ) { return par_( i ); }
      double & operator()( index i ) { return par_[i]; }
      /// The signed transverse curvature (read-only mode)
      double transverseCurvature() const {return par_[ i_tcurv ];}
      /// polar angle  (read-only mode)
      double theta() const {return par_[ i_theta ];}
      /// track azimuthal angle of point of closest approach to beamline (read-only mode)
      double phi0() const { return par_[ i_phi0 ]; }
      /// signed transverse impact parameter (distance of closest approach to beamline) (read-only mode)
      double d0() const { return par_[ i_d0 ]; }
      /// z coordniate of point of closest approach to beamline (read-only mode)
      double dz() const { return par_[ i_dz ]; }
      /// The signed transverse curvature 
      double & transverseCurvature() { return par_[ i_tcurv ]; }
      /// polar angle  
      double & theta() {return par_[ i_theta ];}
      /// track azimuthal angle of point of closest approach to beamline 
      double & phi0() { return par_[ i_phi0 ]; }
      /// signed transverse impact parameter (distance of closest approach to beamline) 
      double & d0() { return par_[ i_d0 ]; }
      /// z coordniate of point of closest approach to beamline 
      double & dz() { return par_[ i_dz ]; }
      /// electric charge
      int charge() const;
      /// transverse momentum
      double pt() const { return pt_; }
      /// momentum vector
      Vector momentum() const;
      /// position of point of closest approach to the beamline
      Point vertex() const;
      /// return a SVector
      ParameterVector vector() const { ParameterVector v; fill( v ); return v; }
      /// fill a SVector
      void fill( ParameterVector & v ) const {
	std::copy( par_.begin(), par_.end(), v.begin() );
      }
    private:
      /// five parameters
      InnerParameterVector par_;
      /// transverse momentum
      Double32_t pt_;
    };

    inline int Parameters::charge() const {
      return transverseCurvature() >0 ? -1 : 1;
    }
    
    inline Vector Parameters::momentum() const {
      return Vector( pt() * cos( phi0() ), pt() * sin( phi0() ), pt() /tan(theta()) );
    }
    
    inline Point Parameters::vertex() const {
      return Point( d0() * sin( phi0() ), - d0() * cos( phi0() ), dz() );
    }

  }
}

#endif
