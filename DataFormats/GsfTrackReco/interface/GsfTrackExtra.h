#ifndef GsfTrackReco_GsfTrackExtra_h
#define GsfTrackReco_GsfTrackExtra_h
/** Extension of a GSF track providing multi-states
 * at the inner- and outermost measurement
 */
// #include "DataFormats/Math/interface/Vector3D.h"
// #include "DataFormats/Math/interface/Point3D.h"
// #include "DataFormats/Math/interface/Vector.h"
// #include "DataFormats/Math/interface/Error.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"

namespace reco {
  class GsfTrackExtra {
  public:
    /// parameter dimension
    enum { dimension = 5 };
//     /// error matrix size
//     enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };
//     /// point in the space
//     typedef math::XYZPoint Point;
//     /// spatial vector
//     typedef math::XYZVector Vector;
//     /// 5 parameter covariance matrix
//     typedef math::Error<dimension>::type CovarianceMatrix;
//     /// index type
//     typedef unsigned int index;
    /// local parameter vector
    typedef math::Vector<dimension>::type LocalParameterVector;
    /// local covariance matrix
    typedef math::Error<dimension>::type LocalCovarianceMatrix;

    /// default constructor
    GsfTrackExtra() { }
    /// constructor from outermost position and momentum
    GsfTrackExtra( const std::vector<GsfComponent5D>& outerStates,
		   const double& outerLocalPzSign, 
		   const std::vector<GsfComponent5D>& innerStates, 
		   const double& innerLocalPzSign);
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
