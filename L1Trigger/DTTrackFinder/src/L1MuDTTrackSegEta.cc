//-------------------------------------------------
//
//   Class: L1MuDTTrackSegEta
//
//   Description: ETA Track Segment 
//
//
//   $Date: 2007/02/27 11:44:00 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP 
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegEta.h"

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

#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"

using namespace std;

// --------------------------------
//       class L1MuDTTrackSegEta
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTTrackSegEta::L1MuDTTrackSegEta() : 
  m_location(), m_position(0), m_quality(0), m_bx(0) {}


L1MuDTTrackSegEta::L1MuDTTrackSegEta(int wheel_id, int sector_id, int station_id, 
                                     int pos, int quality, int  bx) :
  m_location(wheel_id, sector_id, station_id),
  m_position(pos), m_quality(quality), m_bx(bx) {

}


L1MuDTTrackSegEta::L1MuDTTrackSegEta(const L1MuDTTrackSegLoc& id, 
                                     int pos, int quality, int  bx) : 
  m_location(id), m_position(pos), m_quality(quality), m_bx(bx) {

}
  

L1MuDTTrackSegEta::L1MuDTTrackSegEta(const L1MuDTTrackSegEta& id) :
  m_location(id.m_location),
  m_position(id.m_position), m_quality(id.m_quality), m_bx(id.m_bx) {}



//--------------
// Destructor --
//--------------
L1MuDTTrackSegEta::~L1MuDTTrackSegEta() {}


//--------------
// Operations --
//--------------

//
// reset ETA Track Segment
//
void L1MuDTTrackSegEta::reset() {

  m_position = 0;
  m_quality  = 0;
  m_bx       = 0;

}


//
// Assignment operator
//
L1MuDTTrackSegEta& L1MuDTTrackSegEta::operator=(const L1MuDTTrackSegEta& id) {

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
bool L1MuDTTrackSegEta::operator==(const L1MuDTTrackSegEta& id) const {

  if ( m_location != id.m_location ) return false;
  if ( m_position != id.m_position ) return false;
  if ( m_quality  != id.m_quality )  return false;
  if ( m_bx       != id.m_bx )       return false;
  return true;

}


//
// Unequal operator
//
bool L1MuDTTrackSegEta::operator!=(const L1MuDTTrackSegEta& id) const {
  
  if ( m_location != id.m_location ) return true;
  if ( m_position != id.m_position ) return true;
  if ( m_quality  != id.m_quality )  return true;
  if ( m_bx       != id.m_bx )       return true;
  return false;

}


//
// output stream operator
//
ostream& operator<<(ostream& s, const L1MuDTTrackSegEta& id) {

  s.setf(ios::right,ios::adjustfield);
  s << (id.m_location) << "\t" 
    << "position : " << bitset<7>(id.position()) << "  "
    << "quality : " << bitset<7>(id.quality()); 

  return s;

}
