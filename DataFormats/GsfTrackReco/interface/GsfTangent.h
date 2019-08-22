#ifndef GsfTrackReco_GsfTangent_h
#define GsfTrackReco_GsfTangent_h

/** Class holding information on the tangent to the electron track
 *    on one surface
 */

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

namespace reco {

  class GsfTangent {
  public:
    /// default constructor
    GsfTangent() : position_(0., 0., 0.), momentum_(0., 0., 0.), deltaP_(0.), sigDeltaP_(0.) {}
    /// constructor from position, momentum and estimated deltaP
    GsfTangent(const math::XYZPoint& position, const math::XYZVector& momentum, const Measurement1D& deltaP)
        : position_(position), momentum_(momentum) {
      deltaP_ = deltaP.value();
      sigDeltaP_ = deltaP.error();
    }
    const math::XYZPoint& position() const { return position_; }
    const math::XYZVector& momentum() const { return momentum_; }
    /// estimated deltaP (p_out-p_in)
    Measurement1D deltaP() const { return Measurement1D(deltaP_, sigDeltaP_); }

  private:
    math::XYZPoint position_;
    math::XYZVector momentum_;
    double deltaP_;
    double sigDeltaP_;
  };
}  // namespace reco
#endif
