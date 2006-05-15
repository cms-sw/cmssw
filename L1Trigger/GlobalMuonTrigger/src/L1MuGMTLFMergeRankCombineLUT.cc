//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankCombineLUT
//
// 
//   $Date: 2004/02/03 16:33:44 $
//   $Revision: 1.2 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankCombineLUT.h"

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

void L1MuGMTLFMergeRankCombineLUT::InitParameters() {
//  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
};

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMergeRankCombineLUT::TheLookupFunction (int idx, unsigned rank_etaq, unsigned rank_ptq, unsigned rank_etaphi) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  rank_etaq(7) rank_ptq(2) rank_etaphi(1)
  // OUTPUTS: merge_rank(8) 

  return rank_etaq + 128*rank_etaphi;
}; 



















