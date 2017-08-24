//-------------------------------------------------
//
//   Class: L1MuBMDataBuffer
//
//   Description: Data Buffer
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

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMDataBuffer.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorProcessor.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackSegLoc.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"

using namespace std;

// --------------------------------
//       class L1MuBMDataBuffer
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuBMDataBuffer::L1MuBMDataBuffer(const L1MuBMSectorProcessor& sp) :
        m_sp(sp), m_tsphi(nullptr) {


  m_tsphi = new TSPhivector(38);
  m_tsphi->reserve(38);

}


//--------------
// Destructor --
//--------------
L1MuBMDataBuffer::~L1MuBMDataBuffer() {

  delete m_tsphi;

}


//--------------
// Operations --
//--------------

//
// clear buffer
//
void L1MuBMDataBuffer::reset() {

  TSPhivector::iterator iter = m_tsphi->begin();
  while ( iter != m_tsphi->end() ) {
    if ( *iter) {
      delete *iter;
      *iter = nullptr;
    }
    iter++;
  }

}


//
// get phi track segment of a given station
//
const L1MuBMTrackSegPhi* L1MuBMDataBuffer::getTSphi(int station, int reladr) const {

  int address = (station == 1) ? reladr : reladr + (station-2)*12 + 2;
  return (*m_tsphi)[address];

}


//
// add new phi track segment to the buffer
//
void L1MuBMDataBuffer::addTSphi(int adr, const L1MuBMTrackSegPhi& ts) {

  L1MuBMTrackSegPhi* tmpts = new L1MuBMTrackSegPhi(ts);
  (*m_tsphi)[adr] = tmpts;

}


//
// print all phi track segments in the buffer
//
void L1MuBMDataBuffer::printTSphi() const {

  TSPhivector::const_iterator iter = m_tsphi->begin();
  while ( iter != m_tsphi->end() ) {
    if ( *iter ) cout << *(*iter) << endl;
    iter++;
  }

}


//
// count number of non empty phi track segments
//
int L1MuBMDataBuffer::numberTSphi() const {

  int count = 0;
  TSPhivector::iterator iter = m_tsphi->begin();
  while ( iter != m_tsphi->end() ) {
    if ( *iter && !(*iter)->empty() ) count++;
    iter++;
  }
  return count;

}
