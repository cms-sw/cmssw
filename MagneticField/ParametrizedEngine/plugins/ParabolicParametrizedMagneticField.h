#ifndef ParametrizedEngine_ParabolicParametrizedMagneticField_h
#define ParametrizedEngine_ParabolicParametrizedMagneticField_h

/** \class ParabolicParametrizedMagneticField
 *
 *
 *  \author G. Ortona - Torino
 */

#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class TkBfield; }

class ParabolicParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor 
  explicit ParabolicParametrizedMagneticField();

  /// Constructor. Parameters taken from a PSet
  //explicit ParabolicParametrizedMagneticField(const edm::ParameterSet& parameters);

  /// Destructor
  virtual ~ParabolicParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  float B0Z(const float a) const;

  float Kr(const float R2) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  const float b0 = -3.94991e-06;  
  const float b1 = 7.53701e-06;
  const float c1 = 3.8114;     
  const float a  = 2.43878e-11;
};
#endif
