/** \file
 *
 *  $Date: 2010/12/25 16:23:18 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField(){}

MagneticField::~MagneticField(){}


int MagneticField::nominalValue() const {
  return int((inTesla(GlobalPoint(0.f,0.f,0.f))).z() * 10.f + 0.5f);
}
