/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField(){}

MagneticField::~MagneticField(){}


GlobalVector MagneticField::inKGauss(const GlobalPoint& gp) const {
  return inTesla(gp) * 10.;
}

GlobalVector MagneticField::inInverseGeV(const GlobalPoint& gp) const {
  return inTesla(gp) * 2.99792458e-3;
}

int MagneticField::nominalValue() const {
  return int((inTesla(GlobalPoint(0.,0.,0.))).z() * 10. + 0.5);
}
