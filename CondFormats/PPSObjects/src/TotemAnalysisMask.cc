/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"

#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"

//----------------------------------------------------------------------------------------------------

void TotemAnalysisMask::insert(const TotemSymbID& sid, const TotemVFATAnalysisMask& vam) { analysisMask[sid] = vam; }

//----------------------------------------------------------------------------------------------------

void TotemAnalysisMask::print(std::ostream& os) const {
  os << "TotemAnalysisMask mask" << std::endl;

  for (const auto& p : analysisMask) {
    os << "    " << p.first << ": fullMask=" << p.second.fullMask << ", number of masked channels "
       << p.second.maskedChannels.size() << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, TotemAnalysisMask mask) {
  mask.print(os);
  return os;
}

//----------------------------------------------------------------------------------------------------

TYPELOOKUP_DATA_REG(TotemAnalysisMask);
