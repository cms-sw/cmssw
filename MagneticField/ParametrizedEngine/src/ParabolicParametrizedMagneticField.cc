/** \file
 *
 *  \author G. Ortona - Torino
 */

#include "ParabolicParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace std;

ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField()
    : c1(parabolicparametrizedmagneticfield::c1),
      b0(parabolicparametrizedmagneticfield::b0),
      b1(parabolicparametrizedmagneticfield::b1),
      a(parabolicparametrizedmagneticfield::a) {
  setNominalValue();
}

ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField(const vector<double>& parameters)
    : c1(parameters[0]), b0(parameters[1]), b1(parameters[2]), a(parameters[3]) {
  setNominalValue();
}

ParabolicParametrizedMagneticField::~ParabolicParametrizedMagneticField() {}

GlobalVector ParabolicParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {
  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    LogDebug("MagneticField|FieldOutsideValidity")
        << " Point " << gp << " is outside the validity region of ParabolicParametrizedMagneticField";
    return GlobalVector();
  }
}

GlobalVector ParabolicParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  return GlobalVector(0, 0, B0Z(gp.z()) * Kr(gp.perp2()));
}

inline float ParabolicParametrizedMagneticField::B0Z(const float z) const { return b0 * z * z + b1 * z + c1; }

inline float ParabolicParametrizedMagneticField::Kr(const float R2) const { return a * R2 + 1.; }

inline bool ParabolicParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp2() < parabolicparametrizedmagneticfield::tracker_radius2 && fabs(gp.z()) < parabolicparametrizedmagneticfield::tacker_z);
}
