//-------------------------------------------------
//
//   Class: DTSectCollId.cpp
//
//   Description: Definition of sector collectors
//
//
//   Author List:
//   S. Marcellini
//   Modifications: 
//
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSectCollId.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------



#include <iostream>
std::ostream& operator<<(std::ostream &os, const DTSectCollId& id){
  os << "Wheel: " << id.wheel() << " Station: " << id.station() << " Sector: " <<
    id.sector();
  return os;

    id.sector();
}


