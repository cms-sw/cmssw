#ifndef RecoEgamma_EgammaElectronAlgos_interface_FreeTrajectoryState_h
#define RecoEgamma_EgammaElectronAlgos_interface_FreeTrajectoryState_h

#include <cmath>
#include <type_traits>
#include <alpaka/alpaka.hpp>

#include <RecoEgamma/EgammaElectronAlgos/interface/Phys3DVector.h>

namespace egamma {

  // Free trajectory state: position, momentum and charge of a particle
  template <typename T>
  class FreeTrajectoryState {
  public:
    using Vec3D = egamma::math::Phys3DVector<T>;

    // Constructor
    constexpr FreeTrajectoryState(const Vec3D& p, const Vec3D& pos, const int q)
        : momentum_(p), position_(pos), charge_(q) {}

    constexpr Vec3D get_momentum() const { return momentum_; }
    constexpr Vec3D get_position() const { return position_; }
    constexpr int get_charge() const { return charge_; }

  private:
    Vec3D momentum_;    // 3D momentum vector
    Vec3D position_;    // 3D position vector
    const int charge_;  // Particle charge
  };

  // Function to calculate the FreeTrajectoryState from vertex to point
  template <typename TAcc, typename T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE FreeTrajectoryState<T> ftsFromVertexToPoint(
      TAcc const& acc,
      const math::Phys3DVector<T>& xmeas,      // Measured point
      const math::Phys3DVector<T>& xvert,      // Vertex point
      const std::type_identity_t<T> momentum,  // Magnitude of momentum
      const int charge,                        // Charge of the particle
      const std::type_identity_t<T> BInTesla   // Magnetic field (in Tesla)
  ) {
    using Vec3D = math::Phys3DVector<T>;
    //
    // Calculate the difference between measurement and vertex positions
    const Vec3D xdiff = xmeas - xvert;

    // Normalize xdiff and scale by momentum to get the momentum vector
    const T xdiff_mag = xdiff.r(acc);

    // Normalize xdiff and scale by momentum to get the momentum vector:
    const T scale = momentum / xdiff_mag;

    const Vec3D mom = scale * xdiff;

    // Transverse momentum (perpendicular to the z-axis)
    const T pt = mom.rho(acc);
    const T pz = mom[2];

    const T pxOld = mom[0];
    const T pyOld = mom[1];

    // Calculate the curvature (assuming charge is either +1 or -1)
    const T curv = (BInTesla * static_cast<T>(0.29979 * 0.01)) / pt;

    // Calculate the sine and cosine of the rotation angle
    const T sa = static_cast<T>(0.5) * xdiff.rho(acc) * curv * static_cast<T>(charge);
    const T ca = alpaka::math::sqrt(acc, static_cast<T>(1) - sa * sa);

    // Rotate momentum vector in the xy-plane
    const T pxNew = ca * pxOld + sa * pyOld;
    const T pyNew = -sa * pxOld + ca * pyOld;
    //
    const Vec3D pNew(pxNew, pyNew, pz);

    return FreeTrajectoryState<T>(pNew, xmeas, charge);
  }

}  // namespace egamma

#endif  // RecoEgamma_EgammaElectronAlgos_interface_FreeTrajectoryState_h
