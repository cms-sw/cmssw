/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#include "CondFormats/CTPPSReadoutObjects/interface/TotemSymbId.h"

std::ostream& operator << (std::ostream& s, const TotemSymbID &sid)
{
  s << "symb. id=" << sid.symbolicID;

  return s;
}

