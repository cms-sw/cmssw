//-------------------------------------------------
//
//   Class: L1MuBMSecProcId
//
//   Description: Sector Processor identifier
//
//
//
//   Author :
//   N. Neumeister             CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"

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
//       class L1MuBMSecProcId
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMSecProcId::L1MuBMSecProcId() :
      m_wheel(0), m_sector(0) {}

L1MuBMSecProcId::L1MuBMSecProcId(int wheel_id, int sector_id) :
      m_wheel(wheel_id), m_sector(sector_id) {

  if ( !(wheel_id   >= -3 && wheel_id   <=  3) ) {
    //    cerr << "SecProcId : wheel out of range: " << wheel_id << endl;
  }
  if ( !(sector_id  >=  0 && sector_id  <  12) ) {
    //    cerr << "SecProcId : sector out of range: " << sector_id << endl;
  }

}


L1MuBMSecProcId::L1MuBMSecProcId(const L1MuBMSecProcId& id) :
      m_wheel(id.m_wheel), m_sector(id.m_sector) {}


//--------------
// Destructor --
//--------------

L1MuBMSecProcId::~L1MuBMSecProcId() {}

//--------------
// Operations --
//--------------

//
// Assignment operator
//
L1MuBMSecProcId& L1MuBMSecProcId::operator=(const L1MuBMSecProcId& id) {

  if ( this != &id ) {
    m_wheel  = id.m_wheel;
    m_sector = id.m_sector;
  }
  return *this;

}


//
// return logical wheel
//
int L1MuBMSecProcId::locwheel() const {

  return ( m_wheel/abs(m_wheel)*(abs(m_wheel)-1) );

}


//
// Equal operator
//
bool L1MuBMSecProcId::operator==(const L1MuBMSecProcId& id) const {

  if ( wheel()  != id.wheel() )  return false;
  if ( sector() != id.sector() ) return false;
  return true;

}


//
// Unequal operator
//
bool L1MuBMSecProcId::operator!=(const L1MuBMSecProcId& id) const {

  if ( m_wheel  != id.wheel() )  return true;
  if ( m_sector != id.sector() ) return true;
  return false;

}


//
// Less operator
//
bool L1MuBMSecProcId::operator<(const L1MuBMSecProcId& id) const {

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
ostream& operator<<(ostream& s, const L1MuBMSecProcId& id) {

  s.setf(ios::right,ios::adjustfield);
  s << "Sector Processor ( " << setw(2) << id.wheel() << ","
                             << setw(2) << id.sector() << " )";
  return s;

}
