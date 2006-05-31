/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/UniformEngine/src/UniformMagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Vector/interface/GlobalVector.h"


UniformMagneticField::UniformMagneticField(double value)
  : theField(0.,0.,value) {}


GlobalVector UniformMagneticField::inTesla (const GlobalPoint& gp) const {
  return theField;
}
