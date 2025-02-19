//-------------------------------------------------
//
//   Class: L1MuDTTFSetup
//
//   Description: Setup the L1 barrel Muon Trigger Track Finder
//
//
//   $Date: 2007/07/29 17:18:50 $
//   $Revision: 1.4 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFSetup.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"

using namespace std;

// --------------------------------
//       class L1MuDTTFSetup
//---------------------------------


//----------------
// Constructors --
//----------------

L1MuDTTFSetup::L1MuDTTFSetup(const edm::ParameterSet & ps) : m_tf(new L1MuDTTrackFinder(ps)) {

  // setup  the barrel Muon Trigger Track Finder
  m_tf->setup(); 

}


//--------------
// Destructor --
//--------------

L1MuDTTFSetup::~L1MuDTTFSetup() {
  
  delete m_tf;

}


//--------------
// Operations --
//--------------
