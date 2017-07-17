//-------------------------------------------------
//
//   Class: L1MuBMTFSetup
//
//   Description: Setup the L1 barrel Muon Trigger Track Finder
//
//
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTFSetup.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrackFinder.h"

using namespace std;

// --------------------------------
//       class L1MuBMTFSetup
//---------------------------------


//----------------
// Constructors --
//----------------

L1MuBMTFSetup::L1MuBMTFSetup(const edm::ParameterSet & ps, edm::ConsumesCollector && iC) : m_tf(new L1MuBMTrackFinder(ps,std::move(iC))) {
  // setup  the barrel Muon Trigger Track Finder
  m_tf->setup(std::move(iC));

}



//--------------
// Destructor --
//--------------

L1MuBMTFSetup::~L1MuBMTFSetup() {

  delete m_tf;

}



//--------------
// Operations --
//--------------
