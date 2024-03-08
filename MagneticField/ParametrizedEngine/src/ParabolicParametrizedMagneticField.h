#ifndef ParametrizedEngine_ParabolicParametrizedMagneticField_h
#define ParametrizedEngine_ParabolicParametrizedMagneticField_h

/** \class ParabolicParametrizedMagneticField
 *
 *  A simple parametrization of the Bz component in the tracker region
 * using the product of two parabolas
 *
 *  \author G. Ortona - Torino
 */

#include "MagneticField/Engine/interface/MagneticField.h"
#include <vector>

namespace edm {
  class ParameterSet;
}

namespace parabolicparametrizedmagneticfield {
  // Default parameters are the best fit of 3.8T to the OAEParametrizedMagneticField parametrization.
  constexpr float c1 = 3.8114;
  constexpr float b0 = -3.94991e-06;
  constexpr float b1 = 7.53701e-06;
  constexpr float a = 2.43878e-11;
}  // namespace parabolicparametrizedmagneticfield

class ParabolicParametrizedMagneticField final : public MagneticField {
public:
  /// Default constructor, use default values for 3.8T map
  explicit ParabolicParametrizedMagneticField();

  /// Constructor with explicit parameter list (b0, b1, c1, a)
  explicit ParabolicParametrizedMagneticField(const std::vector<double>& parameters);

  /// Destructor
  ~ParabolicParametrizedMagneticField() override;

  GlobalVector inTesla(const GlobalPoint& gp) const override;

  GlobalVector inTeslaUnchecked(const GlobalPoint& gp) const override;

  inline float B0Z(const float a) const;

  inline float Kr(const float R2) const;

  inline bool isDefined(const GlobalPoint& gp) const override;

private:
  float c1;
  float b0;
  float b1;
  float a;
};
#endif
