#ifndef MagneticField_Portable_interface_ParabolicMagneticField_h
#define MagneticField_Portable_interface_ParabolicMagneticField_h

#include <xtd/stdlib/abs.h>

namespace portableParabolicMagneticField {

  struct Parameters {
    // The defaults of ParabolicParametrizedMagneticField: fitted to the 3.8 T field map (160812) over the tracker
    // volume, 0.03 % rms.

    static constexpr float c1 = 3.81036;
    static constexpr float b0 = -2.03767e-06;
    static constexpr float b1 = 7.34495e-06;
    static constexpr float a = 3.01291e-07;
    static constexpr float max_radius2 = 13225.f;  // tracker radius
    static constexpr float max_z = 280.f;          // tracker z
  };

  template <typename Vec3>
  constexpr float Kr(Vec3 const& vec) {
    return Parameters::a * (vec[0] * vec[0] + vec[1] * vec[1]) + 1.f;
  }

  template <typename Vec3>
  constexpr float B0Z(Vec3 const& vec) {
    return Parameters::b0 * vec[2] * vec[2] + Parameters::b1 * vec[2] + Parameters::c1;
  }

  template <typename Vec3>
  constexpr inline bool isValid(Vec3 const& vec) {
    return ((vec[0] * vec[0] + vec[1] * vec[1]) < Parameters::max_radius2 && xtd::abs(vec[2]) < Parameters::max_z);
  }

  template <typename Vec3>
  constexpr inline float magneticFieldAtPoint(Vec3 const& vec) {
    if (isValid(vec)) {
      return B0Z(vec) * Kr(vec);
    } else {
      return 0;
    }
  }

}  // namespace portableParabolicMagneticField
#endif
