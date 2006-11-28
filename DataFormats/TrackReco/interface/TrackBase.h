#ifndef TrackReco_TrackBase_h
#define TrackReco_TrackBase_h
/** \class reco::TrackBase TrackBase.h DataFormats/TrackReco/interface/TrackBase.h
 *
 * Common base class to all track types, including Muon fits.
 * Internally, the following information is stored: <BR>
 *   <DT> Reference position on track: (vx,vy,vz) </DT>
 *   <DT> Momentum at the reference point on track: (px,py,pz) </DT>
 *   <DT> 5D curvilinear covariance matrix from the track fit </DT>
 *   <DT> Charge </DT>
 *   <DT> Chi-square and number of degrees of freedom </DT>
 *   <DT> Summary information of the hit pattern </DT>
 *
 * Parameters associated to the 5D curvilinear covariance matrix: <BR>
 * <B> (qoverp, lambda, phi, d_xy, d_sz) </B><BR>
 * defined as:  <BR>
 *   <DT> qoverp = q / abs(p) = signed inverse of momentum </DT> 
 *   <DT> lambda = pi/2 - polar angle at the given point </DT>
 *   <DT> phi = azimuth angle at the given point </DT>
 *   <DT> d_xy = transverse distance to beam spot (from "straight line" extrapolation if (x,y,z) is not at dca) =  - x * sin(phi) + y * cos(phi) </DT>
 *   <DT> d_sz = distance in SZ plane to beam spot (from "straight line" extrapolation if (x,y,z) is not at dca)  = z * cos(lambda) </DT>
 *
 * according to the definitions given in the following document: <BR>
 * <a href="http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=NOTE&year=2006&files=NOTE2006_001.pdf">A. Strandlie, W. Wittek, "Propagation of Covariance Matrices...", CMS Note 2006/001</a> <BR>
 * 
 * \author Thomas Speer, Luca Lista, Pascal Vanlaer, Juan Alcaraz
 *
 * \version $Id: TrackBase.h,v 1.47 2006/11/27 16:12:39 jalcaraz Exp $
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
    enum { i_qoverp = 0 , i_lambda, i_phi, i_dxy, i_dsz }; 
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
    double qoverp() const { return charge() / p(); }
    /// polar angle  
    double theta() const { return momentum_.theta(); }
    /// Lambda angle
    double lambda() const { return M_PI/2 - momentum_.theta(); }
    /// track impact parameter (distance of closest approach to beamline)
    double dxy() const { return ( - vx() * py() + vy() * px() ) / pt(); }
    /// track impact parameter in perigee convention (d0 = - dxy)
    double d0() const { return - dxy(); }
    /// sz distance to beamline
    double dsz() const { return vz() * pt() / p(); }
    /// z distance to beamline
    double dz() const { return vz(); }
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

    /// track momentum vector
    const Vector & momentum() const { return momentum_; }
    /// position of point of closest approach to the beamline
    const Point & vertex() const { return vertex_; }

    /// dxy with respect to a user-given beamSpot. Use with caution: new beamSpot must be close to (0,0,0), since the extrapolation is linear
    double dxy(const Point& myBeamSpot) const { 
      return ( - (vx()-myBeamSpot.x()) * py() + (vy()-myBeamSpot.y()) * px() ) / pt(); 
    }
    /// dsz with respect to a user-given beamSpot. Use with caution: new beamSpot must be close to (0,0,0), since the extrapolation is linear
    double dsz(const Point& myBeamSpot) const { 
      return ( (vz()-myBeamSpot.z())*pt() - (myBeamSpot.x()*px()+myBeamSpot.y()*py())/pt()*pz() ) / p(); 
    }
    /// dz with respect to a user-given beamSpot. Use with caution: new beamSpot must be close to (0,0,0), since the extrapolation is linear
    double dz(const Point& myBeamSpot) const { 
      return (vz()-myBeamSpot.z()) - (myBeamSpot.x()*px()+myBeamSpot.y()*py())/pt() * (pz()/pt()); 
    }

    /// Parameters with one-to-one corerspondence to the covariance matrix
    ParameterVector parameters() const { 
      return ParameterVector(qoverp(),lambda(),phi(),dxy(),dsz());
    }
    /// return SMatrix
    CovarianceMatrix covariance() const { CovarianceMatrix m; fill( m ); return m; }

    /// i-th parameter ( i = 0, ... 4 )
    double parameter(int i) const { return parameters()[i]; }
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 4 )
    double covariance( int i, int j ) const { return covariance_[ covIndex( i, j ) ]; }
    /// error on specified element
    double error( int i ) const { return sqrt( covariance_[ covIndex( i, i ) ] ); }

    /// error on signed transverse curvature
    double qoverpError() const { return error( i_qoverp ); }
    /// error on theta
    double thetaError() const { return error( i_lambda ); }
    /// error on lambda
    double lambdaError() const { return error( i_lambda ); }
    /// error on eta
    double etaError() const { return error( i_lambda ) / sin(theta()); }
    /// error on phi
    double phiError() const { return error( i_phi ); }
    /// error on dxy
    double dxyError() const { return error( i_dxy ); }
    /// error on d0
    double d0Error() const { return error( i_dxy ); }
    /// error on dsz
    double dszError() const { return error( i_dsz ); }
    /// error on dz
    double dzError() const { return error( i_dsz ) * p()/pt(); }

    /// fill SMatrix
    CovarianceMatrix & fill( CovarianceMatrix & v ) const;
    /// covariance matrix index in array
    static index covIndex( index i, index j );
   
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
