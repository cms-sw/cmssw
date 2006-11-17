//-------------------------------------------------
//
//   Class: L1MuGMTLFDisableHotLUT
//
// 
//   $Date: 2006/05/15 13:56:02 $
//   $Revision: 1.1 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFDisableHotLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
//#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuPacking.h"
//#include "SimG4Core/Notification/interface/Singleton.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFDisableHotLUT::InitParameters() {
//  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
}

//------------------------
// The Lookup Function  --
//------------------------

// This LUT is used to look up whether DT/CSC muons from the other stream
// should be disabled. It is a duplicate of the calculation performed in the other chip. 
// The LUT is needed so that diasbled muons are not considered in the cancel-out process.
// 
// !!! It has to be ensured that the contents match the corresponding LFSortRankEtaPhiLUT !!!
//
//
// If the LFSortRankEtaPhiLUT contains a "11", the LFDisableHotLUT has to contain a '1'
//


unsigned L1MuGMTLFDisableHotLUT::TheLookupFunction (int idx, unsigned eta, unsigned phi) const {
  // idx is DT, CSC
  // INPUTS:  eta(6) phi(8)
  // OUTPUTS: disable_hot(1) 

  // TBD: implementation of reading disable-hot configuration

  return 0;
}



















