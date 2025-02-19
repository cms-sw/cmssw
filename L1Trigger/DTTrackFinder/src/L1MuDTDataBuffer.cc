//-------------------------------------------------
//
//   Class: L1MuDTDataBuffer
//
//   Description: Data Buffer 
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

#include "L1Trigger/DTTrackFinder/src/L1MuDTDataBuffer.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"

using namespace std;

// --------------------------------
//       class L1MuDTDataBuffer
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuDTDataBuffer::L1MuDTDataBuffer(const L1MuDTSectorProcessor& sp) : 
        m_sp(sp), m_tsphi(0) {


  m_tsphi = new TSPhivector(38);
  m_tsphi->reserve(38);

}


//--------------
// Destructor --
//--------------
L1MuDTDataBuffer::~L1MuDTDataBuffer() { 

  delete m_tsphi;
  
}


//--------------
// Operations --
//--------------

//
// clear buffer
//
void L1MuDTDataBuffer::reset() {

  TSPhivector::iterator iter = m_tsphi->begin();
  while ( iter != m_tsphi->end() ) {
    if ( *iter) {
      delete *iter; 
      *iter = 0;
    }
    iter++;
  }

}


//
// get phi track segment of a given station 
//
const L1MuDTTrackSegPhi* L1MuDTDataBuffer::getTSphi(int station, int reladr) const {

  int address = (station == 1) ? reladr : reladr + (station-2)*12 + 2;
  return (*m_tsphi)[address];
  
}


//
// add new phi track segment to the buffer
//
void L1MuDTDataBuffer::addTSphi(int adr, const L1MuDTTrackSegPhi& ts) {

  L1MuDTTrackSegPhi* tmpts = new L1MuDTTrackSegPhi(ts);
  (*m_tsphi)[adr] = tmpts;
  
}


//
// print all phi track segments in the buffer 
//
void L1MuDTDataBuffer::printTSphi() const {

  TSPhivector::const_iterator iter = m_tsphi->begin();
  while ( iter != m_tsphi->end() ) {
    if ( *iter ) cout << *(*iter) << endl;
    iter++;
  }
  
}


//
// count number of non empty phi track segments
//
int L1MuDTDataBuffer::numberTSphi() const {

  int count = 0;
  TSPhivector::iterator iter = m_tsphi->begin();
  while ( iter != m_tsphi->end() ) {
    if ( *iter && !(*iter)->empty() ) count++;
    iter++;
  }
  return count;

}
