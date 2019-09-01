#ifndef GsfTrackReco_GsfTrackExtra_h
#define GsfTrackReco_GsfTrackExtra_h
/** Extension of a GSF track providing multi-states
 * at the inner- and outermost measurement
 */
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
// #include "DataFormats/Math/interface/Error.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"
#include "DataFormats/GsfTrackReco/interface/GsfTangent.h"

#include <iostream>

namespace reco {
  class GsfTrackExtra {
  public:
    /// parameter dimension
    enum { dimension = 5 };
    /// local parameter vector
    typedef math::Vector<dimension>::type LocalParameterVector;
    /// local covariance matrix
    typedef math::Error<dimension>::type LocalCovarianceMatrix;
    /// point in the space
    typedef math::XYZPoint Point;
    /// spatial vector
    typedef math::XYZVector Vector;

    /// default constructor
    GsfTrackExtra() {}
    /// constructor from outermost position and momentum
    GsfTrackExtra(const std::vector<GsfComponent5D>& outerStates,
                  const double& outerLocalPzSign,
                  const std::vector<GsfComponent5D>& innerStates,
                  const double& innerLocalPzSign,
                  const std::vector<GsfTangent>& tangents);
    /// sign of local P_z at outermost state
    double outerStateLocalPzSign() const { return positiveOuterStatePz_ ? 1. : -1.; }
    /// weights at outermost state
    std::vector<double> outerStateWeights() const { return weights(outerStates_); }
    /// local parameters at outermost state
    std::vector<LocalParameterVector> outerStateLocalParameters() const { return parameters(outerStates_); }
    /// local covariance matrices at outermost state
    std::vector<LocalCovarianceMatrix> outerStateCovariances() const { return covariances(outerStates_); }
    /// sign of local P_z at innermost state
    double innerStateLocalPzSign() const { return positiveInnerStatePz_ ? 1. : -1.; }
    /// weights at innermost state
    std::vector<double> innerStateWeights() const { return weights(innerStates_); }
    /// local parameters at innermost state
    std::vector<LocalParameterVector> innerStateLocalParameters() const { return parameters(innerStates_); }
    /// local covariance matrices at innermost state
    std::vector<LocalCovarianceMatrix> innerStateCovariances() const { return covariances(innerStates_); }
    /// number of objects with information for tangents to the electron track
    inline unsigned int tangentsSize() const { return tangents_.size(); }
    /// access to tangent information
    const std::vector<GsfTangent>& tangents() const { return tangents_; }
    /// global position for tangent
    const Point& tangentPosition(unsigned int index) const { return tangents_[index].position(); }
    /// global momentum for tangent
    const Vector& tangentMomentum(unsigned int index) const { return tangents_[index].momentum(); }
    /// deltaP for tangent
    Measurement1D tangentDeltaP(unsigned int index) const { return tangents_[index].deltaP(); }

  private:
    /// extract weights from states
    std::vector<double> weights(const std::vector<GsfComponent5D>& states) const;
    /// extract parameters from states
    std::vector<LocalParameterVector> parameters(const std::vector<GsfComponent5D>& states) const;
    /// extract covariance matrices from states
    std::vector<LocalCovarianceMatrix> covariances(const std::vector<GsfComponent5D>& states) const;

  private:
    /// states at outermost point
    std::vector<GsfComponent5D> outerStates_;
    /// positive sign of P_z(local) at outermost State?
    bool positiveOuterStatePz_;
    /// states at innermost point
    std::vector<GsfComponent5D> innerStates_;
    /// positive sign of P_z(local) at innermost State?
    bool positiveInnerStatePz_;
    /// information for tangents
    std::vector<GsfTangent> tangents_;
  };

}  // namespace reco

#endif
