/** \file
 *
 *  $Revision: 1.2 $
 *  \author P. Janot, copied from N. Amapane, UniformMagneticField - CERN
 */

#include "FastSimulation/TrajectoryManager/interface/LocalMagneticField.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


LocalMagneticField::LocalMagneticField(double value)
  : theField(0.,0.,value) {}


GlobalVector LocalMagneticField::inTesla (const GlobalPoint& gp) const {
  return theField;
}
