//-------------------------------------------------
//
//   Class: DTBtiTrigData.cpp
//
//   Description: DTBtiChip Trigger Data
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>

using namespace std;

//--------------
// Operations --
//--------------

void 
DTBtiTrigData::print() const {
  cout << "BTI Id=" << " ( " << _btiid.wheel()      ;
  cout              << " , " << _btiid.station()    ;
  cout              << " , " << _btiid.sector()     ;
  cout              << " , " << _btiid.superlayer() ;
  cout              << " # " << _btiid.bti()        ;
  cout              << " ) " ;
  cout << ", K=" << K() << ", X=" << X() << ", equation=" << eq();
  cout << ", code=" << code();
  cout << " step= " << step();
/*  cout << " strobe= " << Strobe();
  cout << " Keq values: " << Keq(0) << " " << Keq(1) << " " << Keq(2) << " " 
    << Keq(3) << " " << Keq(4) << " " << Keq(5) << endl;
*/
  cout << endl;
}
