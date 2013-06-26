//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankCombineLUT
//
// 
//   $Date: 2007/04/02 15:45:38 $
//   $Revision: 1.3 $
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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMergeRankCombineLUT::InitParameters() {
}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMergeRankCombineLUT::TheLookupFunction (int idx, unsigned rank_etaq, unsigned rank_ptq, unsigned rank_etaphi) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  rank_etaq(7) rank_ptq(2) rank_etaphi(1)
  // OUTPUTS: merge_rank(8) 

  return rank_etaq + 128*rank_etaphi;
}



















