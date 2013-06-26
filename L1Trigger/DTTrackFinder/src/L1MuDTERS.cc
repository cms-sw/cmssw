//-------------------------------------------------
//
//   Class: L1MuDTERS
//
//   Description: Extrapolation Result Selector
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

#include "L1Trigger/DTTrackFinder/src/L1MuDTERS.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegPhi.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSEU.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTEUX.h"

using namespace std;

// --------------------------------
//       class L1MuDTERS
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTERS::L1MuDTERS(const L1MuDTSEU& seu) : m_seu(seu) {

  reset();

}
  

//--------------
// Destructor --
//--------------

L1MuDTERS::~L1MuDTERS() {}


//--------------
// Operations --
//--------------

//
// run ERS
//
void L1MuDTERS::run() {

  int n_ext = m_seu.numberOfExt();
  if ( n_ext > 0 ) {
    vector<L1MuDTEUX*>::const_iterator first = m_seu.eux().begin();
    vector<L1MuDTEUX*>::const_iterator last  = m_seu.eux().end();
    vector<L1MuDTEUX*>::const_iterator first_max;
    vector<L1MuDTEUX*>::const_iterator second_max;

    // find the best extrapolation
    first_max  = max_element(first, last, L1MuDTEUX::EUX_Comp() );
    m_address[0] = (*first_max)->address();
    m_quality[0] = (*first_max)->quality();
    m_start[0]   = (*first_max)->ts().first;
    m_target[0]  = (*first_max)->ts().second;


    if ( n_ext > 1 ) {
      // find the second best extrapolation 
      second_max = max_element(first, last, L1MuDTEUX::EUX_Comp(*first_max) );
      m_address[1] =  (*second_max)->address();
      m_quality[1] =  (*second_max)->quality();
      m_start[1]   =  (*second_max)->ts().first;
      m_target[1]  =  (*second_max)->ts().second;
    }

    if ( L1MuDTTFConfig::Debug(4) ) {
      cout << "ERS : " << endl;
      cout << "\t first  : " << m_address[0] << '\t' << m_quality[0] << endl;
      cout << "\t second : " << m_address[1] << '\t' << m_quality[1] << endl;
    }

  }

}


//
// reset ERS
//
void L1MuDTERS::reset() {

  for ( int id  = 0; id < 2; id++ ) {
    m_quality[id] = 0; 
    m_address[id] = 15;                                            
    m_start[id]   = 0;
    m_target[id]  = 0;  
  }

}


//
// return pointer to start and target track segment
//
pair<const L1MuDTTrackSegPhi*, const L1MuDTTrackSegPhi*> L1MuDTERS::ts(int id) const {

  return pair<const L1MuDTTrackSegPhi*,const L1MuDTTrackSegPhi*>(m_start[id],m_target[id]);

}
