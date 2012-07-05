/** \file
 *
 *  $Date: 2009/03/19 10:25:35 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField(){}

MagneticField::~MagneticField(){}


int MagneticField::nominalValue() const {
  return int((inTesla(GlobalPoint(0.f,0.f,0.f))).z() * 10.f + 0.5f);
}
