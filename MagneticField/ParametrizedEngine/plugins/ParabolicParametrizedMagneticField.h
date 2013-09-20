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
  float b0 = -0.020789;  
  float b1 = 0.000753701;
  float c1 = 3.8114;     
  float a  = 2.43878e-07;
};
#endif
