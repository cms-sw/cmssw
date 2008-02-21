#ifndef TrackReco_TrackBase_h
#define TrackReco_TrackBase_h
/** \class reco::TrackBase TrackBase.h DataFormats/TrackReco/interface/TrackBase.h
 *
 * Common base class to all track types, including Muon fits.
 * Internally, the following information is stored: <BR>
 *   <DT> A reference position on the track: (vx,vy,vz) </DT>
 *   <DT> Momentum at this given reference point on track: (px,py,pz) </DT>
 *   <DT> 5D curvilinear covariance matrix from the track fit </DT>
 *   <DT> Charge </DT>
 *   <DT> Chi-square and number of degrees of freedom </DT>
 *   <DT> Summary information of the hit pattern </DT>
 *
 * For tracks reconstructed in the CMS Tracker, the reference position is the point of
 * closest approach to the centre of CMS. For muons, this is not necessarily true.
 *
 * Parameters associated to the 5D curvilinear covariance matrix: <BR>
 * <B> (qoverp, lambda, phi, dxy, dsz) </B><BR>
 * defined as:  <BR>
 *   <DT> qoverp = q / abs(p) = signed inverse of momentum [1/GeV] </DT> 
 *   <DT> lambda = pi/2 - polar angle at the given point </DT>
 *   <DT> phi = azimuth angle at the given point </DT>
 *   <DT> dxy = -vx*sin(phi) + vy*cos(phi) [cm] </DT>
 *   <DT> dsz = vz*cos(lambda) - (vx*cos(phi)+vy*sin(phi))*sin(lambda) [cm] </DT>
 *
 * Geometrically, dxy is the signed distance in the XY plane between the
 * the straight line passing through (vx,vy) with azimuthal angle phi and 
 * the point (0,0).<BR>
 * The dsz parameter is the signed distance in the SZ plane between the
 * the straight line passing through (vx,vy,vz) with angles (phi, lambda) and 
 * the point (s=0,z=0). The S axis is defined by the projection of the 
 * straight line onto the XY plane. The convention is to assign the S 
 * coordinate for (vx,vy) as the value vx*cos(phi)+vy*sin(phi). This value is 
 * zero when (vx,vy) is the point of minimum transverse distance to (0,0).
 *
 * Note that dxy and dsz provide sensible estimates of the distance from 
 * the true particle trajectory to (0,0,0) ONLY in two cases:<BR>
 *   <DT> When (vx,vy,vz) already correspond to the point of minimum transverse 
 *   distance to (0,0,0) or is close to it (so that the differences 
 *   between considering the exact trajectory or a straight line in this range 
 *   are negligible). This is usually true for Tracker tracks. </DT>
 *   <DT> When the track has infinite or extremely high momentum </DT>
 *
 * More details about this parametrization are provided in the following document: <BR>
 * <a href="http://cms.cern.ch/iCMS/jsp/openfile.jsp?type=NOTE&year=2006&files=NOTE2006_001.pdf">A. Strandlie, W. Wittek, "Propagation of Covariance Matrices...", CMS Note 2006/001</a> <BR>
 * 
 * \author Thomas Speer, Luca Lista, Pascal Vanlaer, Juan Alcaraz
 *
 * \version $Id: TrackBase.h,v 1.59 2008/01/15 13:05:49 llista Exp $
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
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// enumerator provided indices to the five parameters
    enum { i_qoverp = 0 , i_lambda, i_phi, i_dxy, i_dsz }; 
    /// index type
    typedef unsigned int index;
    
    /// default constructor
    TrackBase();
    /// constructor from fit parameters and error matrix
    TrackBase( double chi2, double ndof, const Point & referencePoint,
	       const Vector & momentum, int charge, const CovarianceMatrix &);
    /// virtual destructor   
    ~TrackBase();
    /// chi-squared of the fit
    double chi2() const { return chi2_; }
    /// number of degrees of freedom of the fit
    double ndof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return chi2_ / ndof_; }
    /// track electric charge
    int charge() const { return charge_; }
    /// q/p 
    double qoverp() const { return charge() / p(); }
    /// polar angle  
    double theta() const { return momentum_.theta(); }
    /// Lambda angle
    double lambda() const { return M_PI/2 - momentum_.theta(); }
    /// dxy parameter. (This is the transverse impact parameter w.r.t. to (0,0,0) ONLY if refPoint is close to (0,0,0): see parametrization definition above for details). See also function dxy(myBeamSpot) below.
    double dxy() const { return ( - vx() * py() + vy() * px() ) / pt(); }
    /// dxy parameter in perigee convention (d0 = - dxy)
    double d0() const { return - dxy(); }
    /// dsz parameter (THIS IS NOT the SZ impact parameter to (0,0,0) if refPoint is far from (0,0,0): see parametrization definition above for details)
    double dsz() const { return vz()*pt()/p() - (vx()*px()+vy()*py())/pt() * pz()/p(); }
    /// dz parameter (= dsz/cos(lambda)). This is the track z0 w.r.t (0,0,0) only if the refPoint is close to (0,0,0). See also function dz(myBeamSpot) below.
    double dz() const { return vz() - (vx()*px()+vy()*py())/pt() * (pz()/pt()); }
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
    /// x coordinate of the reference point on track
    double vx() const { return vertex_.x(); }
    /// y coordinate of the reference point on track
    double vy() const { return vertex_.y(); }
    /// z coordinate of the reference point on track
    double vz() const { return vertex_.z(); }

    /// track momentum vector
    const Vector & momentum() const { return momentum_; }

    /// Reference point on the track
    const Point & referencePoint() const { return vertex_; }

    /// reference point on the track. This method is DEPRECATED, please use referencePoint() instead
    const Point & vertex() const { return vertex_; }

    /// dxy parameter with respect to a user-given beamSpot (WARNING: this quantity can only be interpreted as a minimum transverse distance if beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dxy(const Point& myBeamSpot) const { 
      return ( - (vx()-myBeamSpot.x()) * py() + (vy()-myBeamSpot.y()) * px() ) / pt(); 
    }
    /// dsz parameter with respect to a user-given beamSpot (WARNING: this quantity can only be interpreted as the distance in the S-Z plane to the beamSpot, if the beam spot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dsz(const Point& myBeamSpot) const { 
      return (vz()-myBeamSpot.z())*pt()/p() - ((vx()-myBeamSpot.x())*px()+(vy()-myBeamSpot.y())*py())/pt() *pz()/p(); 
    }
    /// dz parameter with respect to a user-given beamSpot (WARNING: this quantity can only be interpreted as the track z0, if the beamSpot is reasonably close to the refPoint, since linear approximations are involved). This is a good approximation for Tracker tracks.
    double dz(const Point& myBeamSpot) const { 
      return (vz()-myBeamSpot.z()) - ((vx()-myBeamSpot.x())*px()+(vy()-myBeamSpot.y())*py())/pt() * pz()/pt(); 
    }

    /// Track parameters with one-to-one correspondence to the covariance matrix
    ParameterVector parameters() const { 
      return ParameterVector(qoverp(),lambda(),phi(),dxy(),dsz());
    }
    /// return track covariance matrix
    CovarianceMatrix covariance() const { CovarianceMatrix m; fill( m ); return m; }

    /// i-th parameter ( i = 0, ... 4 )
    double parameter(int i) const { return parameters()[i]; }
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 4 )
    double covariance( int i, int j ) const { return covariance_[ covIndex( i, j ) ]; }
    /// error on specified element
    double error( int i ) const { return sqrt( covariance_[ covIndex( i, i ) ] ); }

    /// error on signed transverse curvature
    double qoverpError() const { return error( i_qoverp ); }
    /// error on Pt (set to 1000 TeV if charge==0 for safety)
    double ptError() const { 
      return (charge()!=0) ?  sqrt( 
            pt()*pt()*p()*p()/charge()/charge() * covariance(i_qoverp,i_qoverp)
          + 2*pt()*p()/charge()*pz() * covariance(i_qoverp,i_lambda)
          + pz()*pz() * covariance(i_lambda,i_lambda) ) : 1.e6;
    }
    /// error on theta
    double thetaError() const { return error( i_lambda ); }
    /// error on lambda
    double lambdaError() const { return error( i_lambda ); }
    /// error on eta
    double etaError() const { return error( i_lambda ) * p()/pt(); }
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
   
    ///  Access the hit pattern, indicating in which Tracker layers the track has hits.
    const HitPattern & hitPattern() const { return hitPattern_; }
    /// number of valid hits found 
    unsigned short numberOfValidHits() const { return hitPattern_.numberOfValidHits(); }
    /// number of cases where track crossed a layer without getting a hit.
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
    float chi2_;
    /// number of degrees of freedom
    float ndof_;
     /// innermost (reference) point on track
    Point vertex_;
    /// momentum vector at innermost point
    Vector momentum_;
    /// electric charge
    char charge_;
    /// perigee 5x5 covariance matrix
    float covariance_[ covarianceSize ];
    /// hit pattern
    HitPattern hitPattern_;

  };

  inline TrackBase::index TrackBase::covIndex( index i, index j )  {
    int a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
    return b * ( b + 1 ) / 2 + a;
  }
    
}

#endif
