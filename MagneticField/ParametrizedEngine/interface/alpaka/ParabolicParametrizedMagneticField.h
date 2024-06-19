/**
 Description: Utility function to calculate the Magnetic Field on the GPU
*/

#ifndef MagneticField_ParametrizedEngine_interface_alpaka_ParabolicParametrizedMagneticField_h
#define MagneticField_ParametrizedEngine_interface_alpaka_ParabolicParametrizedMagneticField_h

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace MagneticFieldParabolicPortable {

    struct Parameters {
      float c1 = 3.8114;
      float b0 = -3.94991e-06;
      float b1 = 7.53701e-06;
      float a = 2.43878e-11;
      float max_radius2 = 13225.f;  // tracker radius
      float max_z = 280.f;          // tracker z
    };

    template <typename V3>
    constexpr float Kr(V3 vec) {
      Parameters p;
      return p.a * (vec(0) * vec(0) + vec(1) * vec(1)) + 1.;
    }

    template <typename V3>
    constexpr float B0Z(V3 vec) {
      Parameters p;
      return p.b0 * vec(2) * vec(2) + p.b1 * vec(2) + p.c1;
    }

    template <typename V3>
    constexpr bool isDefined(V3 vec) {
      Parameters p;
      return ((vec(0) * vec(0) + vec(1) * vec(1)) < p.max_radius2 && fabs(vec(2) < p.max_z));
    }

    template <typename V3>
    constexpr float MagneticFieldAtPoint(V3 vec) {
      if (isDefined(vec)) {
        return B0Z(vec) * Kr(vec);
      } else {
        return 0;
      }
    }

  }  // namespace MagneticFieldParabolicPortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
