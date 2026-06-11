#ifndef RecoEgamma_EgammaElectronAlgos_interface_ftsFromVertexToPointPortable_h
#define RecoEgamma_EgammaElectronAlgos_interface_ftsFromVertexToPointPortable_h

#include <cmath>

#include <DataFormats/EgammaReco/interface/alpaka/Phys3DVector.h>

using Vec3d = cms::alpakatools::math::Phys3DVector<double>;

namespace egamma {

  // FreeTrajectoryState template structure
  class FreeTrajectoryState {
  private:
    Vec3d momentum;    // 3D momentum vector
    Vec3d position;    // 3D position vector
    const int charge;  // Particle charge
  public:
    // Constructor
    constexpr FreeTrajectoryState(const Vec3d& p, const Vec3d& pos, const int q)
        : momentum(p), position(pos), charge(q) {}

    constexpr Vec3d get_momentum() const { return momentum; }
    constexpr Vec3d get_position() const { return position; }
    constexpr int get_charge() const { return charge; }
  };

  // Function to calculate the FreeTrajectoryState from vertex to point
  template <typename TAcc>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE FreeTrajectoryState
  ftsFromVertexToPoint(TAcc const& acc,
                       const Vec3d& xmeas,    // Measured point
                       const Vec3d& xvert,    // Vertex point
                       const float momentum,  // Magnitude of momentum
                       const int charge,      // Charge of the particle
                       const float BInTesla   // Magnetic field (in Tesla)
  ) {
    using T = Vec3d::value_type;
    //
    // Calculate the difference between measurement and vertex positions
    const Vec3d xdiff = cms::alpakatools::math::xmy(xmeas, xvert);  //= xmeas - xvert;

    // Normalize xdiff and scale by momentum to get the momentum vector
    const T xdiff_norm = xdiff.norm(acc);

    // Normalize xdiff and scale by momentum to get the momentum vector:
    const T scale = momentum / xdiff_norm;

    const Vec3d mom = cms::alpakatools::math::ax(scale, xdiff);

    // Transverse momentum (perpendicular to the z-axis)
    const T pt = mom.partial_norm(acc);
    const T pz = mom[2];

    const T pxOld = mom[0];
    const T pyOld = mom[1];

    // Calculate the curvature (assuming charge is either +1 or -1)
    const T curv = (BInTesla * 0.29979 * 0.01) / pt;

    // Calculate the sine and cosine of the rotation angle
    const T sa = 0.5 * xdiff.partial_norm(acc) * curv * static_cast<T>(charge);
    const T ca = alpaka::math::sqrt(acc, 1. - sa * sa);

    // Rotate momentum vector in the xy-plane
    const T pxNew = ca * pxOld + sa * pyOld;
    const T pyNew = -sa * pxOld + ca * pyOld;
    //
    const Vec3d pNew(pxNew, pyNew, pz);

    return FreeTrajectoryState(pNew, xmeas, charge);
  }

}  // namespace egamma

#endif  // RecoEgamma_EgammaElectronAlgos_interface_alpaka_ftsFromVertexToPointPortable_h
