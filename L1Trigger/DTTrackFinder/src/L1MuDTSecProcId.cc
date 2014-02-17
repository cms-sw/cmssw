//-------------------------------------------------
//
//   Class: L1MuDTSecProcId
//
//   Description: Sector Processor identifier
//
//
//   $Date: 2008/10/13 07:44:43 $
//   $Revision: 1.3 $
//
//   Author :
//   N. Neumeister             CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cstdlib>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuDTSecProcId
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTSecProcId::L1MuDTSecProcId() :
      m_wheel(0), m_sector(0) {}

L1MuDTSecProcId::L1MuDTSecProcId(int wheel_id, int sector_id) :
      m_wheel(wheel_id), m_sector(sector_id) {

  if ( !(wheel_id   >= -3 && wheel_id   <=  3) ) {
    //    cerr << "SecProcId : wheel out of range: " << wheel_id << endl;
  }
  if ( !(sector_id  >=  0 && sector_id  <  12) ) {
    //    cerr << "SecProcId : sector out of range: " << sector_id << endl;
  }

}


L1MuDTSecProcId::L1MuDTSecProcId(const L1MuDTSecProcId& id) :
      m_wheel(id.m_wheel), m_sector(id.m_sector) {}


//--------------
// Destructor --
//--------------

L1MuDTSecProcId::~L1MuDTSecProcId() {}

//--------------
// Operations --
//--------------

//
// Assignment operator
//
L1MuDTSecProcId& L1MuDTSecProcId::operator=(const L1MuDTSecProcId& id) {

  if ( this != &id ) {
    m_wheel  = id.m_wheel;
    m_sector = id.m_sector;
  }
  return *this;

}


//
// return logical wheel
//
int L1MuDTSecProcId::locwheel() const {

  return ( m_wheel/abs(m_wheel)*(abs(m_wheel)-1) );

}


//
// Equal operator
//
bool L1MuDTSecProcId::operator==(const L1MuDTSecProcId& id) const { 

  if ( wheel()  != id.wheel() )  return false;
  if ( sector() != id.sector() ) return false;
  return true;

}


//
// Unequal operator
//
bool L1MuDTSecProcId::operator!=(const L1MuDTSecProcId& id) const {

  if ( m_wheel  != id.wheel() )  return true;
  if ( m_sector != id.sector() ) return true;
  return false;

}


//
// Less operator
//
bool L1MuDTSecProcId::operator<(const L1MuDTSecProcId& id) const {  

  if ( sector()      < id.sector()     ) return true;
  if ( sector()      > id.sector()     ) return false;
  if ( wheel() < 0 && id.wheel() < 0 ) {
    if ( -wheel() < -id.wheel() ) return true; 
  }
  else {
    if ( wheel() < id.wheel() ) return true;
  }
  return false;

}


//
// output stream operator
//
ostream& operator<<(ostream& s, const L1MuDTSecProcId& id) {

  s.setf(ios::right,ios::adjustfield);
  s << "Sector Processor ( " << setw(2) << id.wheel() << "," 
                             << setw(2) << id.sector() << " )";
  return s;

}
