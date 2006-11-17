//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankEtaQLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaQLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "SimG4Core/Notification/interface/Singleton.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMergeRankEtaQLUT::InitParameters() {
  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMergeRankEtaQLUT::TheLookupFunction (int idx, unsigned eta, unsigned q) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta(6) q(3)
  // OUTPUTS: flag(1) rank_etaq(7) 

  //  int isRPC = idx % 2;
  //  int isFWD = idx / 2;
 
  float etaValue = m_theTriggerScales->getRegionalEtaScale(idx)->getCenter( eta );
  
  
  unsigned flag = 0;
  switch (idx) {
  case 0: // DT
    if ( (q==1 || q==4 || q==6 || q==7) ||
	       ( (q==2 || q==3) && (fabs(etaValue) < 0.9) ) ) flag=1;
    break;
  case 1: // bRPC
    if (q==0) flag=1; 
    break;  
  case 2: // CSC
    if ( (q==2 || fabs(etaValue) < 1.2) || 
	 q==3) flag =1; 
    break; 
  case 3: // fRPC
    if (q==3) flag =1;
  }

  unsigned rank_etaq = 0;

  return flag << 7 | rank_etaq;
}



















