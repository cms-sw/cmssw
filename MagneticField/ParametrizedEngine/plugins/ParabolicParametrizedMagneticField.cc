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
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of ParabolicParametrizedMagneticField";
    return GlobalVector();
  }
}

GlobalVector
ParabolicParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  float x[3] = {gp.x(), gp.y(), gp.z()};
  float B=B0Z(x[2])*Kr(x[0]*x[0]+x[1]*x[1]);
  return GlobalVector(0, 0, B);
}

float ParabolicParametrizedMagneticField::B0Z(const float z) const {
  return b0*z*z + b1*z + c1;
}

float ParabolicParametrizedMagneticField::Kr(const float R2) const {
  return a*R2 +1.;
}

bool
ParabolicParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp2()<(115.f*115.f) && fabs(gp.z())<280.f);
}
