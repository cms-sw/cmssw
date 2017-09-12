//-------------------------------------------------
//
//   Class: L1MuBMTrackSegEta
//
//   Description: ETA Track Segment
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegEta.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <bitset>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackSegLoc.h"

using namespace std;

// --------------------------------
//       class L1MuBMTrackSegEta
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMTrackSegEta::L1MuBMTrackSegEta() :
  m_location(), m_position(0), m_quality(0), m_bx(0) {}


L1MuBMTrackSegEta::L1MuBMTrackSegEta(int wheel_id, int sector_id, int station_id,
                                     int pos, int quality, int  bx) :
  m_location(wheel_id, sector_id, station_id),
  m_position(pos), m_quality(quality), m_bx(bx) {

}


L1MuBMTrackSegEta::L1MuBMTrackSegEta(const L1MuBMTrackSegLoc& id,
                                     int pos, int quality, int  bx) :
  m_location(id), m_position(pos), m_quality(quality), m_bx(bx) {

}


L1MuBMTrackSegEta::L1MuBMTrackSegEta(const L1MuBMTrackSegEta& id) :
  m_location(id.m_location),
  m_position(id.m_position), m_quality(id.m_quality), m_bx(id.m_bx) {}



//--------------
// Destructor --
//--------------
L1MuBMTrackSegEta::~L1MuBMTrackSegEta() {}


//--------------
// Operations --
//--------------

//
// reset ETA Track Segment
//
void L1MuBMTrackSegEta::reset() {

  m_position = 0;
  m_quality  = 0;
  m_bx       = 0;

}


//
// Assignment operator
//
L1MuBMTrackSegEta& L1MuBMTrackSegEta::operator=(const L1MuBMTrackSegEta& id) {

  if ( this != &id ) {
    m_location = id.m_location;
    m_position = id.m_position;
    m_quality  = id.m_quality;
    m_bx       = id.m_bx;
  }
  return *this;

}


//
// Equal operator
//
bool L1MuBMTrackSegEta::operator==(const L1MuBMTrackSegEta& id) const {

  if ( m_location != id.m_location ) return false;
  if ( m_position != id.m_position ) return false;
  if ( m_quality  != id.m_quality )  return false;
  if ( m_bx       != id.m_bx )       return false;
  return true;

}


//
// Unequal operator
//
bool L1MuBMTrackSegEta::operator!=(const L1MuBMTrackSegEta& id) const {

  if ( m_location != id.m_location ) return true;
  if ( m_position != id.m_position ) return true;
  if ( m_quality  != id.m_quality )  return true;
  if ( m_bx       != id.m_bx )       return true;
  return false;

}


//
// output stream operator
//
ostream& operator<<(ostream& s, const L1MuBMTrackSegEta& id) {

  s.setf(ios::right,ios::adjustfield);
  s << (id.m_location) << "\t"
    << "position : " << bitset<7>(id.position()) << "  "
    << "quality : " << bitset<7>(id.quality());

  return s;

}
