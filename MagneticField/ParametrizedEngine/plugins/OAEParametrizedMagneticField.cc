/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "OAEParametrizedMagneticField.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "TkBfield.h"

using namespace std;
using namespace magfieldparam;

OAEParametrizedMagneticField::OAEParametrizedMagneticField(string T) : 
  theParam(T){}


OAEParametrizedMagneticField::OAEParametrizedMagneticField(const edm::ParameterSet& parameters) : 
  theParam(parameters.getParameter<string>("BValue")) {}


OAEParametrizedMagneticField::~OAEParametrizedMagneticField() {}


GlobalVector
OAEParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {
  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of OAEParametrizedMagneticField";
    return GlobalVector();
  }
}

namespace {
  constexpr float ooh = 1./100;
}

GlobalVector
OAEParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  float x[3] = {gp.x()*ooh, gp.y()*ooh, gp.z()*ooh};
  float B[3];
  theParam.getBxyz(x,B);
  return GlobalVector(B[0], B[1], B[2]);
}


bool
OAEParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp2()<(115.f*115.f) && fabs(gp.z())<280.f);
}
