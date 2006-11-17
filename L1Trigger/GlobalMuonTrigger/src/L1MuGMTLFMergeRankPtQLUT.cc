//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankPtQLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankPtQLUT.h"

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

void L1MuGMTLFMergeRankPtQLUT::InitParameters() {
//  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMergeRankPtQLUT::TheLookupFunction (int idx, unsigned q, unsigned pt) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  q(3) pt(5)
  // OUTPUTS: rank_ptq(2) 


  // a dependence of the merge rank on pt an quality can be defined, here
  return 0;
}



















