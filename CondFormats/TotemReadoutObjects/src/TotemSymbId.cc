/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#include "CondFormats/TotemReadoutObjects/interface/TotemSymbId.h"

std::ostream& operator << (std::ostream& s, const TotemSymbID &sid)
{
  switch (sid.subSystem) {
    case TotemSymbID::RP:
      s << "sub-system=RP, ";
      break;
    case TotemSymbID::T1:
      s << "sub-system=T1, ";
      break;
    case TotemSymbID::T2:
      s << "sub-system=T2, ";
      break;
  }

  s << "symb. id=" << sid.symbolicID;

  return s;
}

