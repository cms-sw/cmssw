//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankPtQLUT
//
// 
//   $Date: 2006/11/17 08:25:34 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankPtQLUT.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMergeRankPtQLUT::InitParameters() {
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



















