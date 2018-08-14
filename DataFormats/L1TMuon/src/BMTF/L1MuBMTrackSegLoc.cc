//-------------------------------------------------
//
//   Class: L1MuBMTrackSegLoc
//
//   Description: Logical location of a Track Segment
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

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackSegLoc.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cassert>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuBMTrackSegLoc
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMTrackSegLoc::L1MuBMTrackSegLoc() :
  m_wheel(0), m_sector(0), m_station(0) {}

L1MuBMTrackSegLoc::L1MuBMTrackSegLoc(int wheel_id,
                                     int sector_id,
                                     int station_id) :
  m_wheel(wheel_id), m_sector(sector_id), m_station(station_id) {

  //  assert(wheel_id   >= -3 && wheel_id   <=  3);
  //  assert(sector_id  >=  0 && sector_id  <  12);

  if ( abs(wheel_id) == 3 ) {
    //    assert(station_id >= 3 && station_id <= 4);
  }
  else {
    //    assert(station_id >= 1 && station_id <= 4);
  }

}


L1MuBMTrackSegLoc::L1MuBMTrackSegLoc(const L1MuBMTrackSegLoc &id) :
  m_wheel(id.m_wheel), m_sector(id.m_sector), m_station(id.m_station) {}


//--------------
// Destructor --
//--------------

L1MuBMTrackSegLoc::~L1MuBMTrackSegLoc() {}


//--------------
// Operations --
//--------------

//
// Assignment operator
//
L1MuBMTrackSegLoc& L1MuBMTrackSegLoc::operator=(const L1MuBMTrackSegLoc& id) {

  if ( this != &id ) {
    m_wheel   = id.m_wheel;
    m_sector  = id.m_sector;
    m_station = id.m_station;
  }
  return *this;

}


//
// Equal operator
//
bool L1MuBMTrackSegLoc::operator==(const L1MuBMTrackSegLoc& id) const {

  if ( m_wheel   != id.wheel() )   return false;
  if ( m_sector  != id.sector() )  return false;
  if ( m_station != id.station() ) return false;
  return true;

}


//
// Unequal operator
//
bool L1MuBMTrackSegLoc::operator!=(const L1MuBMTrackSegLoc& id) const {

  if ( m_wheel   != id.wheel() )   return true;
  if ( m_sector  != id.sector() )  return true;
  if ( m_station != id.station() ) return true;
  return false;

}


//
// Less operator
//
bool L1MuBMTrackSegLoc::operator<(const L1MuBMTrackSegLoc& id) const {

  if ( wheel()   < id.wheel()   ) return true;
  if ( wheel()   > id.wheel()   ) return false;
  if ( sector()  < id.sector()  ) return true;
  if ( sector()  > id.sector()  ) return false;
  if ( station() < id.station() ) return true;
  if ( station() > id.station() ) return false;
  return false;

}


//
// output stream operator
//
ostream& operator<<(ostream& s, const L1MuBMTrackSegLoc& id) {

  s.setf(ios::right,ios::adjustfield);
  s << "wheel = "   << setw(2) << id.wheel()   << "  "
    << "sector = "  << setw(2) << id.sector()  << "  "
    << "station = " << setw(1) << id.station();

  return s;

}
