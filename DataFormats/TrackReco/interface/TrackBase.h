#ifndef TrackReco_TrackBase_h
#define TrackReco_TrackBase_h
/** \class reco::TrackBase TrackBase.h DataFormats/TrackReco/interface/TrackBase.h
 *
 * Common base class to all track types, including Muon fits.
 * It provides fit parameters in cartesian representation
 * and covariance matrix in perigee parametrization, chi-square and 
 * and summary information of the hit pattern. Transverse momentum is
 * also stored to avoid access to magnetic field.
 *
 * Model of 5 perigee parameters for Track fit:<BR>
 * <B> (kappa, theta, phi_0, d_0, z_0) </B><BR>
 * defined as:  <BR>
 *   <DT> kappa = q / p_T = signed transverse curvature </DT> 
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
 * \version $Id: TrackBase.h,v 1.43 2006/10/31 19:23:28 llista Exp $
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
    /// error matrix size
    enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };
    /// parameter vector
    typedef math::Vector<dimension>::type ParameterVector;
    /// 5 parameter covariance matrix
    typedef math::Error<dimension>::type CovarianceMatrix;
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
    TrackBase( double chi2, double ndof, const Point & , const Vector &, int charge,
	       const CovarianceMatrix &);
   
    /// chi-squared of the fit
    double chi2() const { return chi2_; }
    /// number of degrees of freedom of the fit
    double ndof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return chi2_ / ndof_; }
    /// track electric charge
    int charge() const { return charge_; }
    /// transverse curvature
    double transverseCurvature() const { return charge() / pt(); }
    /// track azimutal angle at vertex
    double phi0() const { return momentum_.phi(); }
    /// polar angle  
    double theta() const { return momentum_.theta(); }
    /// track impact parameter (distance of closest approach to beamline)
    double d0() const {  return ( vx() * py() - vy() * px() ) / p (); }
    /// z coordniate of point of closest approach to beamline
    double dz() const { return vz() * pz() / p(); }

    /// track momentum vector
    const Vector & momentum() const { return momentum_; }
    /// position of point of closest approach to the beamline
    const Point & vertex() const { return vertex_; }
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 4 )

    double covariance( int i, int j ) const { return covariance_[ covIndex( i, j ) ]; }
    /// error on specified element
    double error( int i ) const { return sqrt( covariance_[ covIndex( i, i ) ] ); }
    /// error on signed transverse curvature
    double transverseCurvatureError() const { return error( i_transverseCurvature ); }
    /// error on theta
    double thetaError() const { return error( i_theta ); }
    /// error on phi0
    double phi0Error() const { return error( i_phi0 ); }
    /// error on d0
    double d0Error() const { return error( i_d0 ); }
    /// error on dx
    double dzError() const { return error( i_dz ); }
    /// return SMatrix
    CovarianceMatrix covariance() const { CovarianceMatrix m; fill( m ); return m; }
    /// fill SMatrix
    CovarianceMatrix & fill( CovarianceMatrix & v ) const;
    /// covariance matrix index in array
    static index covIndex( index i, index j );
   
    /// momentum vector magnitude
    double p() const { return momentum_.R(); }
    /// track transverse momentum
    double pt() const { return sqrt( momentum_.Perp2() ); }
    /// x coordinate of momentum vector
    double px() const { return momentum_.x(); }
    /// y coordinate of momentum vector
    double py() const { return momentum_.y(); }
    /// z coordinate of momentum vector
    double pz() const { return momentum_.z(); }
    /// azimuthal angle of momentum vector
    double phi() const { return momentum_.Phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return momentum_.Eta(); }
    /// x coordinate of point of closest approach to the beamline
    double vx() const { return vertex_.x(); }
    /// y coordinate of point of closest approach to the beamline
    double vy() const { return vertex_.y(); }
    /// z coordinate of point of closest approach to the beamline
    double vz() const { return vertex_.z(); }
    
    //  hit pattern
    const HitPattern & hitPattern() const { return hitPattern_; }
    /// number of hits found 
    unsigned short numberOfValidHits() const { return hitPattern_.numberOfValidHits(); }
    /// number of hits lost
    unsigned short numberOfLostHits() const { return hitPattern_.numberOfLostHits(); }
    /// set hit pattern from vector of hit references
    template<typename C>
    void setHitPattern( const C & c ) { hitPattern_.set( c.begin(), c.end() ); }
    template<typename I>
    void setHitPattern( const I & begin, const I & end ) { hitPattern_.set( begin, end ); }
    /// set hit pattern for specified hit
    void setHitPattern( const TrackingRecHit & hit, size_t i ) { hitPattern_.set( hit, i ); }
    /// position index 

  private:
    /// chi-squared
    Double32_t chi2_;
    /// number of degrees of freedom
    Double32_t ndof_;
     /// innermost point
    Point vertex_;
    /// momentum vector at innermost point
    Vector momentum_;
    /// electric charge
    char charge_;
    /// transverse momentum
    Double32_t pt_;
    /// perigee 5x5 covariance matrix
    Double32_t covariance_[ covarianceSize ];
    /// hit pattern
    HitPattern hitPattern_;

  };

  inline TrackBase::index TrackBase::covIndex( index i, index j )  {
    int a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
    return b * ( b + 1 ) / 2 + a;
  }
    
}

#endif
