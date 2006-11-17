//-------------------------------------------------
//
//   Class: L1MuGMTMIAUEtaConvLUT
//
// 
//   $Date: 2006/08/21 14:23:13 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaConvLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTScales.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuPacking.h"
#include "SimG4Core/Notification/interface/Singleton.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTMIAUEtaConvLUT::InitParameters() {
  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  m_theGMTScales = Singleton<L1MuGMTScales>::instance();
}

//--------------------------------------------------------------------------------
// Eta conversion LUT: converts 6-bit input eta to 4 bits
// ===================
// 
// Because the phi projection LUT 1 can only accept 4 bits of eta information,
// the eta-conversion LUT converts from the (non-linear) input scales to 
// different 4-bit scales (for DT, CSC, BRPC, FRPC).
//
// The 4-bit eta is coded as a symmteric scale with pseudo-sign.
// For the scale see GMTScales::ReducedEtaScale()
//
// In the HW this LUT is implemented as asynchronous distributed RAM.
//
//--------------------------------------------------------------------------------

unsigned L1MuGMTMIAUEtaConvLUT::TheLookupFunction (int idx, unsigned eta_in) const {
  // idx is MIP_DT, MIP_BRPC, ISO_DT, ISO_BRPC, MIP_CSC, MIP_FRPC, ISO_CSC, ISO_FRPC
  // INPUTS:  eta_in(6)
  // OUTPUTS: eta_out(4) 

  int isRPC = idx % 2;
  int isFWD = idx / 4;

  int idx_drcr = isFWD * 2 + isRPC;
  
  float etaValue = m_theTriggerScales->getRegionalEtaScale(idx_drcr)->getCenter( eta_in );

  unsigned eta4bit = 0;
  if ( (isRPC && isFWD && fabs(etaValue) < m_theGMTScales->getReducedEtaScale(3)->getScaleMin() ) ||
       (isRPC && !isFWD && fabs(etaValue) > m_theGMTScales->getReducedEtaScale(1)->getScaleMax() )) {
    if(!m_saveFlag) edm::LogWarning("LUTRangeViolation") 
       << "L1MuGMTMIAUEtaConvLUT::TheLookupFunction: RPC " << (isFWD?"fwd":"brl") << " eta value out of range: " << etaValue << endl;
  }
  else 
    eta4bit = m_theGMTScales->getReducedEtaScale(idx_drcr)->getPacked( etaValue );
  //  cout << "etaValue  = " << etaValue << "   eta4bit= " << eta4bit << endl;

  return eta4bit;
}














