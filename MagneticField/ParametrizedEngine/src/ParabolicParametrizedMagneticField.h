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

namespace edm { class ParameterSet; }

class ParabolicParametrizedMagneticField : public MagneticField {
 public:
  /// Default constructor, use default values for 3.8T map
  explicit ParabolicParametrizedMagneticField();

  /// Constructor with explicit parameter list (b0, b1, c1, a)
  explicit ParabolicParametrizedMagneticField(const std::vector<double>& parameters);

  /// Destructor
  virtual ~ParabolicParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  inline float B0Z(const float a) const;

  inline float Kr(const float R2) const;

  inline bool isDefined(const GlobalPoint& gp) const;

 private:
  float c1;
  float b0;
  float b1;
  float a;
};
#endif
