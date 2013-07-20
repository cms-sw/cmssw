/** \file
 *
 *  $Date: 2011/12/10 16:03:31 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField() : nominalValueCompiuted(false){}

MagneticField::~MagneticField(){}


int MagneticField::computeNominalValue() const {
  return int((inTesla(GlobalPoint(0.f,0.f,0.f))).z() * 10.f + 0.5f);
}
