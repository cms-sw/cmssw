//-------------------------------------------------
//
//   Class: L1MuDTSEU
//
//   Description: Single Extrapolation Unit
//
//
//   $Date: 2008/11/28 10:30:51 $
//   $Revision: 1.4 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTSEU.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTDataBuffer.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTExtrapolationUnit.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTEUX.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTERS.h"

using namespace std;

// --------------------------------
//       class L1MuDTSEU
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTSEU::L1MuDTSEU(const L1MuDTSectorProcessor& sp, Extrapolation ext, unsigned int tsId) : 
     m_sp(sp), m_ext(ext), 
     m_startTS_Id(tsId), m_startTS(0), m_EUXs(), m_ERS() {
 
  m_EUXs.reserve(12);

  for ( int target_ts = 0; target_ts < 12; target_ts++ ) {
    m_EUXs.push_back( new L1MuDTEUX(m_sp,*this,target_ts) );
  }

  m_ERS = new L1MuDTERS(*this);

}


//--------------
// Destructor --
//--------------

L1MuDTSEU::~L1MuDTSEU() {

  vector<L1MuDTEUX*>::iterator iter_eux;
  for ( iter_eux = m_EUXs.begin(); iter_eux != m_EUXs.end(); iter_eux++ ) {
    delete (*iter_eux);
    *iter_eux = 0;
  }

  m_startTS = 0;
  m_EUXs.clear();

  delete m_ERS;

}


//--------------
// Operations --
//--------------

//
// run SEU
//
void L1MuDTSEU::run(const edm::EventSetup& c) {

  if ( L1MuDTTFConfig::Debug(3) ) cout << "Run SEU " << m_ext << " " 
                                       << m_startTS_Id << endl;

  pair<int,int> ext_pair = L1MuDTExtrapolationUnit::which_ext(m_ext);
  int target = ext_pair.second;

  // check if it is a nextWheel or ownWheel SEU
  bool nextWheel = isNextWheelSEU();

  // relative addresses used
  //                                         nextWheel
  //           extrapolation               extrapolation
  //              address                    address
  //        +--------+--------+         +--------+--------+
  //    +   |        |        |     +   |        |        |
  //        |  4   5 |  6   7 |         |        |  6   7 |
  //    |   |        |        |     |   |        |        |
  //    |   +--------+--------+     |   +--------+--------+
  //        |........|        |         |........|        |
  //   phi  |..0...1.|  2   3 |    phi  |........|  2   3 |
  //        |........|        |         |........|        |
  //    |   +--------+--------+     |   +--------+--------+
  //    |   |        |        |     |   |        |        |
  //        |  8   9 | 10  11 |         |        | 10  11 |
  //    -   |        |        |     -   |        |        |
  //        +--------+--------+         +--------+--------+
  //
  //             -- eta --                   -- eta --  

  // loop over all 12 target addresses
  for ( int reladr = 0; reladr < 12; reladr++ ) {

    // for the nextWheel extrapolations only reladr: 2,3,6,7,10,11 
    if ( nextWheel && (reladr/2)%2 == 0 ) continue;

    const L1MuDTTrackSegPhi* target_ts = m_sp.data()->getTSphi(target, reladr);
    if ( target_ts && !target_ts->empty() ) {
      m_EUXs[reladr]->load(m_startTS, target_ts);
      m_EUXs[reladr]->run(c);
      if ( m_EUXs[reladr]->result() ) m_EXtable.set(reladr);
    }

  }

  if ( L1MuDTTFConfig::Debug(3) ) {
    int n_ext = numberOfExt();
    if ( n_ext > 0 ) cout << "number of successful EUX : " <<  n_ext << endl;
  }

  if ( m_ERS ) m_ERS->run();

  //  if ( m_ERS->address(0) != 15 ) m_QStable.set(m_ERS->address(0));
  //  if ( m_ERS->address(1) != 15 ) m_QStable.set(m_ERS->address(1));
  m_QStable = m_EXtable;

}


//
// reset SEU
//
void L1MuDTSEU::reset() {

  m_startTS = 0;
  vector<L1MuDTEUX*>::iterator iter_eux;
  for ( iter_eux = m_EUXs.begin(); iter_eux != m_EUXs.end(); iter_eux++ ) {
    (*iter_eux)->reset();
  }

  m_ERS->reset();
  
  m_EXtable.reset();
  m_QStable.reset();
  
}


//
// reset a single extrapolation
//
void L1MuDTSEU::reset(unsigned int relAdr) {

  m_EXtable.reset(relAdr);
  m_QStable.reset(relAdr);
  m_EUXs[relAdr]->reset();
//  m_ERS->reset();
  
}


//
// get number of successful extrapolations
//
int L1MuDTSEU::numberOfExt() const {

  int number = 0;
  vector<L1MuDTEUX*>::const_iterator iter_eux;
  for ( iter_eux = m_EUXs.begin(); iter_eux != m_EUXs.end(); iter_eux++ ) {
    if ( (*iter_eux)->result() ) number++;
  }

  return number;

}
