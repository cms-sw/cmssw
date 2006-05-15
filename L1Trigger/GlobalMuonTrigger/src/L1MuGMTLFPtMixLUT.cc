//-------------------------------------------------
//
//   Class: L1MuGMTLFPtMixLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPtMixLUT.h"

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

void L1MuGMTLFPtMixLUT::InitParameters() {
//  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
};

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFPtMixLUT::TheLookupFunction (int idx, unsigned pt_dtcsc, unsigned pt_rpc) const {
  // idx is DTRPC, CSCRPC
  // INPUTS:  pt_dtcsc(5) pt_rpc(5)
  // OUTPUTS: pt_mixed(5) 


  // implement minimum by default
  
  return pt_dtcsc > pt_rpc ? pt_rpc : pt_dtcsc;
}; 



















