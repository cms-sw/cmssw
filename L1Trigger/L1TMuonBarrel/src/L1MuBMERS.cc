//-------------------------------------------------
//
//   Class: L1MuBMERS
//
//   Description: Extrapolation Result Selector
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

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMERS.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSEU.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMEUX.h"

using namespace std;

// --------------------------------
//       class L1MuBMERS
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMERS::L1MuBMERS(const L1MuBMSEU& seu) : m_seu(seu) {

  reset();

}


//--------------
// Destructor --
//--------------

L1MuBMERS::~L1MuBMERS() {}


//--------------
// Operations --
//--------------

//
// run ERS
//
void L1MuBMERS::run() {

  int n_ext = m_seu.numberOfExt();
  if ( n_ext > 0 ) {
    vector<L1MuBMEUX*>::const_iterator first = m_seu.eux().begin();
    vector<L1MuBMEUX*>::const_iterator last  = m_seu.eux().end();
    vector<L1MuBMEUX*>::const_iterator first_max;
    vector<L1MuBMEUX*>::const_iterator second_max;

    // find the best extrapolation
    first_max  = max_element(first, last, L1MuBMEUX::EUX_Comp() );
    m_address[0] = (*first_max)->address();
    m_quality[0] = (*first_max)->quality();
    m_start[0]   = (*first_max)->ts().first;
    m_target[0]  = (*first_max)->ts().second;


    if ( n_ext > 1 ) {
      // find the second best extrapolation
      second_max = max_element(first, last, L1MuBMEUX::EUX_Comp(*first_max) );
      m_address[1] =  (*second_max)->address();
      m_quality[1] =  (*second_max)->quality();
      m_start[1]   =  (*second_max)->ts().first;
      m_target[1]  =  (*second_max)->ts().second;
    }

    if ( L1MuBMTFConfig::Debug(4) ) {
      cout << "ERS : " << endl;
      cout << "\t first  : " << m_address[0] << '\t' << m_quality[0] << endl;
      cout << "\t second : " << m_address[1] << '\t' << m_quality[1] << endl;
    }

  }

}


//
// reset ERS
//
void L1MuBMERS::reset() {

  for ( int id  = 0; id < 2; id++ ) {
    m_quality[id] = 0;
    m_address[id] = 15;
    m_start[id]   = nullptr;
    m_target[id]  = nullptr;
  }

}


//
// return pointer to start and target track segment
//
pair<const L1MuBMTrackSegPhi*, const L1MuBMTrackSegPhi*> L1MuBMERS::ts(int id) const {

  return pair<const L1MuBMTrackSegPhi*,const L1MuBMTrackSegPhi*>(m_start[id],m_target[id]);

}
