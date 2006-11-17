//-------------------------------------------------
//
//   Class: L1MuGMTLFSortRankCombineLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankCombineLUT.h"

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

void L1MuGMTLFSortRankCombineLUT::InitParameters() {
//  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
}

//--------------------------------------------------------------------------------
// Sort Rank Combination LUT
//
// This LUT combines the three contributioons to the sort-rank.
// It can be used to lower the rank of or to disable muons in certain
// hot detector regions
//
// Inputs:  rank_etaq(2-bit), rank_ptq(7-bit), rank_etaphi(2-bit)
// Outputs: Sort Rank (8-bit) 
//
//--------------------------------------------------------------------------------

unsigned L1MuGMTLFSortRankCombineLUT::TheLookupFunction (int idx, unsigned rank_etaq, unsigned rank_ptq, unsigned rank_etaphi) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  rank_etaq(2) rank_ptq(7) rank_etaphi(2)
  // OUTPUTS: sort_rank(8) 

  //  int isRPC = idx % 2;
  //  int isFWD = idx / 2;

  // by default return maximum
  unsigned int rank_combined = rank_etaphi * 32 + rank_ptq;
  // max is 127 + 32*2 
  if (rank_etaphi == 3) rank_combined = 0;
 
  return rank_combined;
}



















