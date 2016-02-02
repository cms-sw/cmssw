//-------------------------------------------------
//
//   Description: Parameters for Assignment
//
//
//   $Date: 2007/02/27 11:44:00 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMAssParam.h"
#include <iostream>

using namespace std;

//
// overload output stream operator for PtAssMethod
//
ostream& operator<<( ostream& s, PtAssMethod method) {

  switch (method) {
    case PT12L  : { return s << "PT12L "; break; }
    case PT12H  : { return s << "PT12H "; break; }
    case PT13L  : { return s << "PT13L "; break; }
    case PT13H  : { return s << "PT13H "; break; }
    case PT14L  : { return s << "PT14L "; break; }
    case PT14H  : { return s << "PT14H "; break; }
    case PT23L  : { return s << "PT23L "; break; }
    case PT23H  : { return s << "PT23H "; break; }
    case PT24L  : { return s << "PT24L "; break; }
    case PT24H  : { return s << "PT24H "; break; }
    case PT34L  : { return s << "PT34L "; break; }
    case PT34H  : { return s << "PT34H "; break; }

    default :
      return s << "unknown pt-assignment method ";
  }

}
