//-------------------------------------------------
//
//   Class: L1MuGMTLFSortRankEtaPhiLUT
//
// 
//   $Date: 2007/04/02 15:45:39 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankEtaPhiLUT.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFSortRankEtaPhiLUT::InitParameters() {
}

//--------------------------------------------------------------------------------
// Sort Rank LUT, Eta-Phi part
//
// This LUT determines the dependency of the sort rank on Eta and Phi. 
// It can be used to lower the rank of or to disable muons in certain
// hot detector regions
//
// Inputs:  eta(6 bit) and phi(8 bit)
// Outputs: Rank contribution 2-bit
//
//
//
// PROPOSAL FOR PROGRAMMING
//
// 00	Reduce below all other: rank += 0
// 01 	Reduce by half scale:	rank += 64
// 10 	Normal operation:	rank += 128
// 11 	Switch off:	        rank = 0
//
// Switched off muons (code 11 binary) will be disabled completely, also for the matchiing
// by an additional disable-signal.
//
//--------------------------------------------------------------------------------

unsigned L1MuGMTLFSortRankEtaPhiLUT::TheLookupFunction (int idx, unsigned eta, unsigned phi) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta(6) phi(8)
  // OUTPUTS: rank_etaphi(2) 

  // by default return code 10 (binary)
  unsigned int rank_etaphi = 2;
  return rank_etaphi;
}



















