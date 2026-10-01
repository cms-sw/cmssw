/** \file
 *
 *  \author G. Ortona - Torino
 */

#include "ParabolicParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

using namespace std;

// Default parameters are fitted to the CMS field map (160812, the default MagneticField) over the tracker volume,
// |z| < 280 cm and r < 115 cm: 0.03 % rms.  The previous constants, the best fit of 3.8T to the
// OAEParametrizedMagneticField parametrization {3.8114, -3.94991e-06, 7.53701e-06, 2.43878e-11}, are low by
// 1.46 % on average against that map, and by up to 4.7 % at the ends of the tracker.
ParabolicParametrizedMagneticField::ParabolicParametrizedMagneticField()
    : c1(3.81036), b0(-2.03767e-06), b1(7.34495e-06), a(3.01291e-07) {
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
  return (gp.perp2() < (13225.f) && fabs(gp.z()) < 280.f);
}
