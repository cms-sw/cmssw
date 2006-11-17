//-------------------------------------------------
//
//   Class: L1MuGMTLFSortRankEtaQLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankEtaQLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "SimG4Core/Notification/interface/Singleton.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFSortRankEtaQLUT::InitParameters() {
  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
}

//------------------------------------------------------------------------------------
// Sort Rank LUT, Eta-Quality part 
// 
// This LUT determines the dependency of the sort rank on Eta and Quality. 
// Inputs:  Eta(6 bit) and Quality(3 bit)
// Outputs: Very-low-quality bits (2 bit)
//          Rank contribution 2-bit
//
// The very-low quality bits are used to suppress unmatched muons in certain 
// trigger-algorithms. By default vlq(1)(MSB) means: not to be used in di-muon trigger
// and vlq(0)(LSB) means not to be used in single-muon trigger. 
//
// The Rank-contribution from Eta-Quality is currently not used. Its influence on the
// sort-rank can be determined in the Sort-Rank Combine LUT.
//
//------------------------------------------------------------------------------------

unsigned L1MuGMTLFSortRankEtaQLUT::TheLookupFunction (int idx, unsigned eta, unsigned q) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta(6) q(3)
  // OUTPUTS: vlq(2) rank_etaq(2) 
  int isRPC = idx % 2;
  //  int isFWD = idx / 2;

  float etaValue = fabs ( m_theTriggerScales->getRegionalEtaScale(idx)->getCenter( eta ) );

  //
  // very-low-quality flags
  //
  // 0 .. no VLQ set
  // 1 .. output muon will be quality 2 (disable in single and di-muon trigger)
  // 2 .. output muon will be quality 3 (disable in single-muon trigger only)
  // 3 .. output muon will be quality 4 (disable in di-muon trigger only)

  unsigned vlq = 0;
  // RPC selection
  if (isRPC) {
    if ( ( q == 0 && ( ( etaValue > 1.04 && etaValue < 1.24 ) || // Q0, high rate, high noise
		       ( etaValue > 1.48 ) ) ) ||                // Q0, high rate, high noise
	 ( q == 1 && ( ( etaValue > 0.83 && etaValue < 1.04 ) || // Q1, high rate
		       ( etaValue > 1.14 && etaValue < 1.24 ) || // Q1, high noise
		       ( etaValue > 1.36 ) ) ) ||                // Q1, high rate
	 ( q == 2 && ( etaValue > 0.83 && etaValue < 0.93 ) ) )  // Q2, high rate
      vlq = 1;
  }                         
      
  // CSC selection
  if ( idx == 2 ) { // CSC
    if (q == 2) {
      //      if ( etaValue > 1.1 && etaValue < 1.2 ) 
      //vlq = 1; // disable in single and di-muon trigger
      //else 
      vlq = 2; // disable in single-muon trigger only
    }
    if (q == 1) {
      if (etaValue > 1.2)
	vlq = 2;   // disable in single-muon trigger only
    }
  }
    
  if ( L1MuGMTConfig::getDoOvlRpcAnd() ) {
    if ( idx == 0 ) { // DT
      if ( etaValue > 0.91 ) vlq = 1;
    }
    if ( idx == 2 ) { // CSC
      if ( etaValue <  1.06 ) vlq = 1;  
    }
  }

  //
  // Rank contribution from eta and quality
  //

  // by default return maximum 
  // LUT can later be used to reduce the sort rank of certain regions
  unsigned rank_etaq = 3;

  return (vlq << 2)  | rank_etaq;
}



















