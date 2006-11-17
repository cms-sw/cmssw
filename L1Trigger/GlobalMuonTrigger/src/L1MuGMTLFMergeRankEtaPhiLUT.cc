//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankEtaPhiLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaPhiLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
//#include "SimG4Core/Notification/interface/Singleton.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMergeRankEtaPhiLUT::InitParameters() {
//  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMergeRankEtaPhiLUT::TheLookupFunction (int idx, unsigned eta, unsigned phi) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta(6) phi(8)
  // OUTPUTS: rank_etaphi(1) 

  // return zero to reduce merge rank for a certain region
  return 1;
}



















