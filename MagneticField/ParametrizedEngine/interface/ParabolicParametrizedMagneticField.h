/**
 Description: Utility function to calculate the Magnetic Field on the GPU. The Vec3 argument of the functions must support access to its components via (), note that e.g. Eigen::Matrix provides such an interface.
*/

#ifndef MagneticField_ParametrizedEngine_interface_ParabolicParametrizedMagneticField_h
#define MagneticField_ParametrizedEngine_interface_ParabolicParametrizedMagneticField_h

namespace magneticFieldParabolicPortable {

  struct Parameters {
    // These parameters are the best fit of 3.8T to the OAEParametrizedMagneticField parametrization.
    // See MagneticField/ParametrizedEngine/src/ParabolicParametrizedMagneticField.cc
    static constexpr float c1 = 3.8114;
    static constexpr float b0 = -3.94991e-06;
    static constexpr float b1 = 7.53701e-06;
    static constexpr float a = 2.43878e-11;
    static constexpr float max_radius2 = 13225.f;  // tracker radius
    static constexpr float max_z = 280.f;          // tracker z
  };

  template <typename Vec3>
  constexpr float Kr(Vec3 const& vec) {
    return Parameters::a * (vec(0) * vec(0) + vec(1) * vec(1)) + 1.f;
  }

  template <typename Vec3>
  constexpr float B0Z(Vec3 const& vec) {
    return Parameters::b0 * vec(2) * vec(2) + Parameters::b1 * vec(2) + Parameters::c1;
  }

  template <typename Vec3>
  constexpr bool isValid(Vec3 const& vec) {
    return ((vec(0) * vec(0) + vec(1) * vec(1)) < Parameters::max_radius2 && fabs(vec(2)) < Parameters::max_z);
  }

  template <typename Vec3>
  constexpr float magneticFieldAtPoint(Vec3 const& vec) {
    if (isValid(vec)) {
      return B0Z(vec) * Kr(vec);
    } else {
      return 0;
    }
  }

}  // namespace magneticFieldParabolicPortable

#endif
