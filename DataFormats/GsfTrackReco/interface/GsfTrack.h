#ifndef GsfTrackReco_GsfTrack_h
#define GsfTrackReco_GsfTrack_h
/** Extension of reco::Track for GSF. It contains
 * one additional Ref to a GsfTrackExtra object.
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h" 
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace reco {

  class GsfTrack : public Track {
  public:
    /// parameter dimension mode
    enum { dimensionMode = 3 };
    /// error matrix size mode
    enum { covarianceSizeMode = dimensionMode * ( dimensionMode + 1 ) / 2 };
    /// parameter vector (momentum part) from mode
    typedef math::Vector<dimensionMode>::type ParameterVectorMode;
    /// 3 parameter covariance matrix (momentum part) from mode
    typedef math::Error<dimensionMode>::type CovarianceMatrixMode;
    /// default constructor
    GsfTrack();
    /// constructor from fit parameters and error matrix
    /// notice that the reference point must be 
    /// the point of closest approch to the beamline.    
    GsfTrack( double chi2, double ndof, const Point &, const Vector &, int charge,
	      const CovarianceMatrix & );
    /// set reference to GSF "extra" object
    void setGsfExtra( const GsfTrackExtraRef & ref ) { gsfExtra_ = ref; }
    /// reference to "extra" object
    const GsfTrackExtraRef & gsfExtra() const { return gsfExtra_; }

    /// set mode parameters
    void setMode (int chargeMode, const Vector& momentumMode,
		  const CovarianceMatrixMode& covarianceMode);

    /// track electric charge from mode
    int chargeMode() const { return chargeMode_; }
    /// q/p  from mode
    double qoverpMode() const { return chargeMode() / pMode(); }
    /// polar angle   from mode
    double thetaMode() const { return momentumMode_.theta(); }
    /// Lambda angle from mode
    double lambdaMode() const { return M_PI/2 - momentumMode_.theta(); }
    /// momentum vector magnitude from mode
    double pMode() const { return momentumMode_.R(); }
    /// track transverse momentum from mode
    double ptMode() const { return sqrt( momentumMode_.Perp2() ); }
    /// x coordinate of momentum vector from mode
    double pxMode() const { return momentumMode_.x(); }
    /// y coordinate of momentum vector from mode
    double pyMode() const { return momentumMode_.y(); }
    /// z coordinate of momentum vector from mode
    double pzMode() const { return momentumMode_.z(); }
    /// azimuthal angle of momentum vector from mode
    double phiMode() const { return momentumMode_.Phi(); }
    /// pseudorapidity of momentum vector from mode
    double etaMode() const { return momentumMode_.Eta(); }

    /// track momentum vector from mode
    const Vector & momentumMode() const { return momentumMode_; }

    /// Track parameters with one-to-one correspondence to the covariance matrix from mode
    ParameterVectorMode parametersMode() const { 
      return ParameterVectorMode(qoverpMode(),lambdaMode(),phiMode());
    }
    /// return track covariance matrix from mode
    CovarianceMatrixMode covarianceMode() const { CovarianceMatrixMode m; fill( m ); return m; }

    /// i-th parameter ( i = 0, ... 2 ) from mode
    double parameterMode(int i) const { return parametersMode()[i]; }
    /// (i,j)-th element of covarianve matrix ( i, j = 0, ... 2 ) from mode
    double covarianceMode( int i, int j ) const { return covarianceMode_[ covIndex( i, j ) ]; }
    /// error on specified element from mode
    double errorMode( int i ) const { return sqrt( covarianceMode_[ covIndex( i, i ) ] ); }

    /// error on signed transverse curvature from mode
    double qoverpModeError() const { return errorMode( i_qoverp ); }
    /// error on Pt (set to 1000 TeV if charge==0 for safety) from mode
    double ptModeError() const { 
      return (chargeMode()!=0) ?  sqrt( 
            ptMode()*ptMode()*pMode()*pMode()/chargeMode()/chargeMode() * covarianceMode(i_qoverp,i_qoverp)
          + 2*ptMode()*pMode()/chargeMode()*pzMode() * covarianceMode(i_qoverp,i_lambda)
          + pzMode()*pzMode() * covarianceMode(i_lambda,i_lambda) ) : 1.e6;
    }
    /// error on theta from mode
    double thetaModeError() const { return errorMode( i_lambda ); }
    /// error on lambda from mode
    double lambdaModeError() const { return errorMode( i_lambda ); }
    /// error on eta from mode
    double etaModeError() const { return errorMode( i_lambda ) * pMode()/ptMode(); }
    /// error on phi from mode
    double phiModeError() const { return errorMode( i_phi ); }

  private:
    /// fill 3x3 SMatrix
    CovarianceMatrixMode & fill CMS_THREAD_SAFE ( CovarianceMatrixMode & v ) const;


  private:
    /// reference to GSF "extra" extension
    GsfTrackExtraRef gsfExtra_;
    /// electric charge from mode
    char chargeMode_;
    /// momentum vector from mode
    Vector momentumMode_;
    /// 3x3 momentum part of covariance (in q/p, lambda, phi)
    float covarianceMode_[ covarianceSizeMode ];

  };

}

#endif
