/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#include "CondFormats/PPSObjects/interface/TotemSymbId.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

std::ostream& operator<<(std::ostream& s, const TotemSymbID& sid) {
  s << "DetId=" << sid.symbolicID << " (" << CTPPSDetId(sid.symbolicID) << ")";

  return s;
}
