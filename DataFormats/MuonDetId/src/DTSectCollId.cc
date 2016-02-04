//-------------------------------------------------
//
//   Class: DTSectCollId.cc
//
//   Description: Definition of sector collectors
//
//
//   Author List:
//   S. Marcellini
//   Modifications:
//   11/12/06 C.Battilana : new SectCollId definitions 
//
//
//--------------------------------------------------


//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------



#include <iostream>
std::ostream& operator<<(std::ostream &os, const DTSectCollId& id){
  os << "Wheel: "   << id.wheel() 
     << " Sector: " << id.sector();
  return os;
}


