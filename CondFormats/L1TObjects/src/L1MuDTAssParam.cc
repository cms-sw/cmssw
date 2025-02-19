//-------------------------------------------------
//
//   Description: Parameters for Assignment
//
//
//   $Date: 2007/03/30 07:48:02 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTAssParam.h"
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
    case PT12LO : { return s << "PT12LO "; break; }
    case PT12HO : { return s << "PT12HO "; break; }    
    case PT13LO : { return s << "PT13LO "; break; }
    case PT13HO : { return s << "PT13HO "; break; }    
    case PT14LO : { return s << "PT14LO "; break; }
    case PT14HO : { return s << "PT14HO "; break; }    
    case PT23LO : { return s << "PT23LO "; break; }
    case PT23HO : { return s << "PT23HO "; break; }
    case PT24LO : { return s << "PT24LO "; break; }
    case PT24HO : { return s << "PT24HO "; break; }
    case PT34LO : { return s << "PT34LO "; break; }
    case PT34HO : { return s << "PT34HO "; break; }
    case PT15LO : { return s << "PT15LO "; break; }
    case PT15HO : { return s << "PT15HO "; break; }
    case PT25LO : { return s << "PT25LO "; break; }
    case PT25HO : { return s << "PT25HO "; break; }
    default :
      return s << "unknown pt-assignment method ";
  }

}
