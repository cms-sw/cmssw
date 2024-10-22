/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include <sstream>

MagVolumeOutsideValidity::MagVolumeOutsideValidity(MagVolume::LocalPoint l, MagVolume::LocalPoint u) throw()
    : lower_(l), upper_(u) {
  std::stringstream linestr;
  linestr << "Magnetic field requested outside of validity of the MagVolume: " << lower() << " - " << upper()
          << std::endl;
  m_message = linestr.str();
}
