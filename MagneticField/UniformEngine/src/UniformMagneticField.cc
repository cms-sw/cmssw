/** \file
 *
 *  $Date: 2007/02/03 16:04:09 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/UniformEngine/src/UniformMagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


UniformMagneticField::UniformMagneticField(double value)
  : theField(0.,0.,value) {}


GlobalVector UniformMagneticField::inTesla (const GlobalPoint& gp) const {
  return theField;
}

GlobalVector UniformMagneticField::inTeslaUnchecked (const GlobalPoint& gp) const {
  return theField;
}
