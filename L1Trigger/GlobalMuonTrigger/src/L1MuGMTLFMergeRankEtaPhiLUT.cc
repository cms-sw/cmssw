//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankEtaPhiLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaPhiLUT.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMergeRankEtaPhiLUT::InitParameters() {
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



















