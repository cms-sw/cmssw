/** \file
 *
 *  \author G. Ortona - Torino
 */

#include "ParabolicParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace std;


// Default parameters are the best fit of 3.8T to the OAEParametrizedMagneticField parametrization.
ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField() :
  c1(3.8114),
  b0(-3.94991e-06),
  b1(7.53701e-06),
  a (2.43878e-11)
{}


ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField(const vector<double>& parameters) :
  c1(parameters[0]),
  b0(parameters[1]),
  b1(parameters[2]),
  a (parameters[3])
{}


ParabolicParametrizedMagneticField::~ParabolicParametrizedMagneticField() {}


GlobalVector
ParabolicParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {
  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    LogDebug("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of ParabolicParametrizedMagneticField";
    return GlobalVector();
  }
}

GlobalVector ParabolicParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  return GlobalVector(0, 0, B0Z(gp.z())*Kr(gp.perp2()));
}

inline float ParabolicParametrizedMagneticField::B0Z(const float z) const {
  return b0*z*z + b1*z + c1;
}

inline float ParabolicParametrizedMagneticField::Kr(const float R2) const {
  return a*R2 +1.;
}

inline bool ParabolicParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp2()<(13225.f) && fabs(gp.z())<280.f);
}
