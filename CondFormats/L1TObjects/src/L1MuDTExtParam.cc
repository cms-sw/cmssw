//-------------------------------------------------
//
//   Description: Parameters for Extrapolation
//
//
//   $Date: 2007/03/30 07:48:02 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troocniz              UAM Madrid
//
//--------------------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
#include <iostream>

using namespace std;

//
// output stream operator for Extrapolation
//
ostream& operator<<( ostream& s, Extrapolation ext) {

  switch ( ext ) {
    case EX12: return s << "EX12 ";
    case EX13: return s << "EX13 ";
    case EX14: return s << "EX14 ";
    case EX21: return s << "EX21 ";
    case EX23: return s << "EX23 ";
    case EX24: return s << "EX24 ";
    case EX34: return s << "EX34 ";
    case EX15: return s << "EX15 ";
    case EX25: return s << "EX25 ";
    case EX16: return s << "EX16 ";
    case EX26: return s << "EX26 ";
    case EX56: return s << "EX56 ";
    default: return s << "unknown extrapolation ";
  }

}
