#ifndef TrackReco_GsfTrackExtra_h
#define TrackReco_GsfTrackExtra_h
/** Extension of a GSF track, based on TrackExtra.
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/TrackExtraBase.h"
#include "DataFormats/TrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/GsfComponent5D.h"

namespace reco {
  class GsfTrackExtra : public TrackExtraBase {
  public:
    /// parameter dimension
    enum { dimension = 5 };
    /// error matrix size
    enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };
    /// point in the space
    typedef math::XYZPoint Point;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// 5 parameter covariance matrix
    typedef math::Error<dimension>::type CovarianceMatrix;
    /// index type
    typedef unsigned int index;
    /// local parameter vector
    typedef math::Vector<dimension>::type LocalParameterVector;
    /// local covariance matrix
    typedef math::Error<dimension>::type LocalCovarianceMatrix;

    /// default constructor
    GsfTrackExtra() { }
    /// constructor from outermost position and momentum
    GsfTrackExtra( const Point & outerPosition, const Vector & outerMomentum, 
		   const CovarianceMatrix& outerCov,
		   const std::vector<GsfComponent5D>& outerStates,
		   const double& outerLocalPzSign, unsigned int outerId, bool ok,
		   const Point & innerPosition, const Vector & innerMomentum,
		   const CovarianceMatrix& innerCov,
		   const std::vector<GsfComponent5D>& innerStates, 
		   const double& innerLocalPzSign, unsigned int innerId, bool iok);
    /// outermost point
    const Point & outerPosition() const { return outerPosition_; }
    /// momentum vector at outermost point
    const Vector & outerMomentum() const { return outerMomentum_; }
    /// returns true if the outermost point is valid
    bool outerOk() const { return outerOk_; }
    /// innermost point
    const Point & innerPosition() const { return innerPosition_; }
    /// momentum vector at innermost point
    const Vector & innerMomentum() const { return innerMomentum_; }
    /// returns true if the innermost point is valid
    bool innerOk() const { return innerOk_; }
    /// x coordinate of momentum vector at the outermost point    
    double outerPx() const { return outerMomentum_.X(); }
    /// y coordinate of momentum vector at the outermost point
    double outerPy() const { return outerMomentum_.Y(); }
    /// z coordinate of momentum vector at the outermost point
    double outerPz() const { return outerMomentum_.Z(); }
    /// x coordinate the outermost point
    double outerX() const { return outerPosition_.X(); }
    /// y coordinate the outermost point
    double outerY() const { return outerPosition_.Y(); }
    /// z coordinate the outermost point
    double outerZ() const { return outerPosition_.Z(); }
    /// magnitude of momentum vector at the outermost point    
    double outerP() const { return outerMomentum().R(); }
    /// transverse momentum at the outermost point    
    double outerPt() const { return outerMomentum().Rho(); }
    /// azimuthal angle of the  momentum vector at the outermost point
    double outerPhi() const { return outerMomentum().Phi(); }
    /// pseudorapidity the  momentum vector at the outermost point
    double outerEta() const { return outerMomentum().Eta(); }
    /// polar angle of the  momentum vector at the outermost point
    double outerTheta() const { return outerMomentum().Theta(); }
    /// polar radius of the outermost point
    double outerRadius() const { return outerPosition().Rho(); }

    /// outermost trajectory state curvilinear errors
    CovarianceMatrix outerStateCovariance() const;
    /// innermost trajectory state curvilinear errors
    CovarianceMatrix innerStateCovariance() const;
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillOuter( CovarianceMatrix & v ) const;
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillInner( CovarianceMatrix & v ) const;
    /// DetId of the detector on which surface the outermost state is located
    unsigned int outerDetId() const { return outerDetId_; }
    /// DetId of the detector on which surface the innermost state is located
    unsigned int innerDetId() const { return innerDetId_; }

    /// sign of local P_z at outermost state
    double outerStateLocalPzSign() const {return positiveOuterStatePz_ ? 1. : -1.;}
    /// weights at outermost state
    std::vector<double> outerStateWeights() const { return weights(outerStates_); }
    /// local parameters at outermost state
    std::vector<LocalParameterVector> outerStateLocalParameters() const { 
      return parameters(outerStates_); 
    }
    /// local covariance matrices at outermost state
    std::vector<LocalCovarianceMatrix> outerStateCovariances() const {
      return covariances(outerStates_);
    }
    /// sign of local P_z at innermost state
    double innerStateLocalPzSign() const {return positiveInnerStatePz_ ? 1. : -1.;}
    /// weights at innermost state
    std::vector<double> innerStateWeights() const { return weights(innerStates_); }
    /// local parameters at innermost state
    std::vector<LocalParameterVector> innerStateLocalParameters() const { 
      return parameters(innerStates_); 
    }
    /// local covariance matrices at innermost state
    std::vector<LocalCovarianceMatrix> innerStateCovariances() const {
      return covariances(innerStates_);
    }

  private:
    /// extract weights from states
    std::vector<double> weights (const std::vector<GsfComponent5D>& states) const;
    /// extract parameters from states
    std::vector<LocalParameterVector> parameters (const std::vector<GsfComponent5D>& states) const;
    /// extract covariance matrices from states
    std::vector<LocalCovarianceMatrix> covariances (const std::vector<GsfComponent5D>& states) const;
    
  private:
    /// outermost point
    Point outerPosition_;
    /// momentum vector at outermost point
    Vector outerMomentum_;
    /// outermost point validity flag
    bool outerOk_;
    /// outermost trajectory state curvilinear errors 
    Double32_t outerCovariance_[ covarianceSize ];
    unsigned int outerDetId_;


    /// innermost point
    Point innerPosition_;
    /// momentum vector at innermost point
    Vector innerMomentum_;
    /// innermost point validity flag
    bool innerOk_;
    /// innermost trajectory state 
    Double32_t innerCovariance_[ covarianceSize ];
    unsigned int innerDetId_;

    /// states at outermost point
    std::vector<GsfComponent5D> outerStates_;
    /// positive sign of P_z(local) at outermost State?
    bool positiveOuterStatePz_;
    /// states at innermost point
    std::vector<GsfComponent5D> innerStates_;
    /// positive sign of P_z(local) at innermost State?
    bool positiveInnerStatePz_;
  };

}

#endif
