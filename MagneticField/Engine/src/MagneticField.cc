/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField() = default;

MagneticField::MagneticField(const MagneticField& orig) = default;

MagneticField::~MagneticField() = default;

void MagneticField::setNominalValue() {
  auto const at0z = inTesla(GlobalPoint(0.f, 0.f, 0.f)).z();
  theNominalValue = int(at0z * 10.f + 0.5f);
  theInverseBzAtOriginInGeV = 1.f / (at0z * 2.99792458e-3f);
}
