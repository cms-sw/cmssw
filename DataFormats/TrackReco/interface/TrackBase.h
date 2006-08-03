#ifndef TrackReco_TrackBase_h
#define TrackReco_TrackBase_h
/** \class reco::TrackBase TrackBase.h DataFormats/TrackReco/interface/TrackBase.h
 *
 * Common base class to all track types, including Muon fits.
 * It provides fit parameters and covariance matrix, chi-square and 
 * and summary information of the hit pattern. Transverse momentum is
 * also stored to avoid access to magnetic field.
 *
 * Model of 5 perigee parameters for Track fit:<BR>
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
 * \author Thomas Speer, Luca Lista, Pascal Vanlaer
 *
 * \version $Id: TrackBase.h,v 1.27 2006/08/02 10:58:34 llista Exp $
 *
 */

#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"

namespace reco {

  class TrackBase {
  public:
    /// parameter dimension
    enum { dimension = 5 };
    /// parameter vector
    typedef math::Vector<dimension>::type ParameterVector;
    /// 5 parameter covariance matrix
    typedef math::Error<dimension>::type CovarianceMatrix;
    /// matrix size
    enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };
    /// position-momentum covariance matrix (6x6)
    typedef math::Error<6>::type PosMomError;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// enumerator provided indices to the five parameters
    enum { i_transverseCurvature = 0 , i_theta, i_phi0, i_d0, i_dz }; 
     /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// index type
    typedef unsigned int index;
    
    /// default constructor
    TrackBase() { }
    /// constructor from fit parameters and error matrix
    TrackBase( double chi2, double ndof,
	       const ParameterVector & par, double pt, const CovarianceMatrix & cov );
    /// set hit pattern from vector of hit references
    void setHitPattern( const TrackingRecHitRefVector & hitlist ) {
      hitPattern_.set( hitlist );
    }
   
    /// chi-squared of the fit
    double chi2() const { return chi2_; }
    /// number of degrees of freedom of the fit
    double ndof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return chi2_ / ndof_; }
    /// i-th fit parameter ( i = 0, ... 4 )
    double & parameter( int i ) { return parameters_[ i ]; }
    /// i-th fit parameter ( i = 0, ... 4 )
    const double & parameter( int i ) const { return parameters_[ i ]; }
    /// track electric charge
    int charge() const { return transverseCurvature() > 0 ? -1 : 1; }
    /// The signed transverse curvature
    double transverseCurvature() const { return parameters_[ i_transverseCurvature ]; }
    /// track azimutal angle of point of closest approach to beamline
    double phi0() const { return parameters_[ i_phi0 ]; }
    /// polar angle  
    double theta() const { return parameters_[ i_theta ]; }
    /// track impact parameter (distance of closest approach to beamline)
    double d0() const { return parameters_[ i_d0 ]; }
    /// z coordniate of point of closest approach to beamline
    double dz() const { return parameters_[ i_dz ]; }
    /// track momentum vector
    Vector momentum() const;
    /// position of point of closest approach to the beamline
    Point vertex() const;
    /// return a SVector
    ParameterVector parameters() const { ParameterVector v; fill( v ); return v; }
    /// fill a SVector
    ParameterVector & fill( ParameterVector & v ) const;
    
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 4 )
    double & covariance( int i, int j ) { return covariance_[ idx( i, j ) ]; }
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 4 )
    const double & covariance( int i, int j ) const { return covariance_[ idx( i, j ) ]; }
    /// error on specified element
    double error( int i ) const { return sqrt( covariance_[ idx( i, i ) ] ); }
    
    /// error on signed transverse curvature
    double transverseCurvatureError() const { return covariance_[ idx( i_transverseCurvature, i_transverseCurvature ) ]; }
    /// error on theta
    double thetaError() const { return covariance_[ idx( i_theta, i_theta ) ]; }
    /// error on phi0
    double phi0Error() const { return covariance_[ idx ( i_phi0, i_phi0 ) ]; }
    /// error on d0
    double d0Error() const { return covariance_[ idx( i_d0, i_d0 ) ]; }
    /// error on dx
    double dzError() const { return covariance_[ idx( i_dz, i_dz ) ]; }
    /// return SMatrix
    CovarianceMatrix covariance() const { CovarianceMatrix m; fill( m ); return m; }
    /// fill SMatrix
    CovarianceMatrix & fill( CovarianceMatrix & v ) const;
    
    /// momentum vector magnitude
    double p() const { return momentum().R(); }
    /// track transverse momentum
    double pt() const { return pt_; }
    /// x coordinate of momentum vector
    double px() const { return momentum().X(); }
    /// y coordinate of momentum vector
    double py() const { return momentum().Y(); }
    /// z coordinate of momentum vector
    double pz() const { return momentum().Z(); }
    /// azimuthal angle of momentum vector
    double phi() const { return momentum().Phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return momentum().Eta(); }
    /// x coordinate of point of closest approach to the beamline
    double x() const { return vertex().X(); }
    /// y coordinate of point of closest approach to the beamline
    double y() const { return vertex().Y(); }
    /// z coordinate of point of closest approach to the beamline
    double z() const { return vertex().Z(); }
    
    //  hit pattern
    const HitPattern & hitPattern() const { return hitPattern_; }
    /// number of hits found 
    unsigned short numberOfValidHits() const { return hitPattern_.numberOfValidHits(); }
    /// number of hits lost
    unsigned short numberOfLostHits() const { return hitPattern_.numberOfLostHits(); }
    
  private:
    /// chi-squared
    Double32_t chi2_;
    /// number of degrees of freedom
    Double32_t ndof_;
    // perigee 5 parameters
    Double32_t parameters_[ dimension ];
    /// transverse momentum
    Double32_t pt_;
    /// perigee 5x5 covariance matrix
    Double32_t covariance_[ covarianceSize ];
    /// hit pattern
    HitPattern hitPattern_;
    /// position index
    index idx( index i, index j ) const {
      int a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
      return a * dimension + b - a * ( a + 1 ) / 2;
    }
  };
  
  inline TrackBase::Vector TrackBase::momentum() const {
    return Vector( pt() * cos( phi0() ), pt() * sin( phi0() ), pt() / tan( theta() ) );
  }
  
  inline TrackBase::Point TrackBase::vertex() const {
    return Point( d0() * sin( phi0() ), - d0() * cos( phi0() ), dz() );
  }
    
}

#endif
