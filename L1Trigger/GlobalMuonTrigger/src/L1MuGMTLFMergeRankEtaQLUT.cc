//-------------------------------------------------
//
//   Class: L1MuGMTLFMergeRankEtaQLUT
//
// 
//   $Date: 2012/03/14 13:52:09 $
//   $Revision: 1.8 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMergeRankEtaQLUT::InitParameters() {
}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMergeRankEtaQLUT::TheLookupFunction (int idx, unsigned eta, unsigned q) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta(6) q(3)
  // OUTPUTS: flag(1) rank_etaq(7) 

  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();

  //  int isRPC = idx % 2;
  //  int isFWD = idx / 2;
 
  float etaValue = theTriggerScales->getRegionalEtaScale(idx)->getCenter( eta );
  
  
  unsigned flag = 0;
  switch (idx) {
  case 0: // DT
//    if ( (q==1 || q==4 || q==6 || q==7) ||
//	       ( (q==2 || q==3) && (fabs(etaValue) < 0.9) ) ) flag=1;
	flag =1;
    break;
  case 1: // bRPC
    if (q==0 and fabs(etaValue)>0.7) flag=1;
//    flag =1;
    break;  
  case 2: // CSC
//    if ( (q==2 || fabs(etaValue) < 1.2) ||  q==3) flag =1;
    if (q==3) flag =1;
    break; 
  case 3: // fRPC
//    if (q==3) flag =1;
	flag =1;
  }

  // use local quality as rank
  unsigned rank_etaq = q;
  // add 1 to RPC in order to promote it in case of equality (should go with the fix in L1MuGMTMerger.cc)
  if( idx==1 || idx==3 ) rank_etaq++;
  if(m_GeneralLUTVersion == 0) {
	  // in the overlap region promote RPC (valid for 2011 data)
	  if( (idx==1 || idx==3) && (fabs(etaValue)>1. && fabs(etaValue)<1.23) ) rank_etaq=7;
  }

  return flag << 7 | rank_etaq;
}



















