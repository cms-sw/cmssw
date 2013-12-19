/** \file
 *
 *  \author G. Ortona - Torino
 */

#include "ParabolicParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace std;
using namespace magfieldparam;

ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField() {}


//ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField(const edm::ParameterSet& parameters) {}
//  theParam(parameters.getParameter<string>("BValue")) {}


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
  float B=B0Z(gp.z())*Kr(gp.perp2());
  return GlobalVector(0, 0, B);
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
