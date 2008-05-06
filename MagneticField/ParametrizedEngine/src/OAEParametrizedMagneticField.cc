/** \file
 *
 *  $Date: 2008/04/24 14:57:28 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - CERN
 */

#include <MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "TkBfield.h"

using namespace std;
using namespace magfieldparam;

OAEParametrizedMagneticField::OAEParametrizedMagneticField(string & T) : 
  theParam(new TkBfield(T))
{}


OAEParametrizedMagneticField::OAEParametrizedMagneticField(const edm::ParameterSet& parameters) {
  theParam = new TkBfield(parameters.getParameter<string>("BValue"));
}


OAEParametrizedMagneticField::~OAEParametrizedMagneticField() {
  delete theParam;
}


GlobalVector
OAEParametrizedMagneticField::inTesla(const GlobalPoint& gp) const {
  if (isDefined(gp)) {
    return inTeslaUnchecked(gp);
  } else {
    edm::LogWarning("MagneticField|FieldOutsideValidity") << " Point " << gp << " is outside the validity region of OAEParametrizedMagneticField";
    return GlobalVector();
  }
}

GlobalVector
OAEParametrizedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  double x[3] = {gp.x()/100., gp.y()/100., gp.z()/100.};
  double B[3];
  theParam->getBxyz(x,B);
  return GlobalVector(B[0], B[1], B[2]);
}


bool
OAEParametrizedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (gp.perp()<115. && fabs(gp.z())<280.);
}
