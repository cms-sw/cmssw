//-------------------------------------------------
//
//   Class: L1MuGMTMIAUEtaConvLUT
//
// 
//   $Date: 2007/04/02 15:45:39 $
//   $Revision: 1.6 $
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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTMIAUEtaConvLUT::InitParameters() {
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

  const L1MuGMTScales* theGMTScales = L1MuGMTConfig::getGMTScales();
  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();

  int isRPC = idx % 2;
  int isFWD = idx / 4;

  int idx_drcr = isFWD * 2 + isRPC;
  
  float etaValue = theTriggerScales->getRegionalEtaScale(idx_drcr)->getCenter( eta_in );

  unsigned eta4bit = 0;
  if ( (isRPC && isFWD && fabs(etaValue) < theGMTScales->getReducedEtaScale(3)->getScaleMin() ) ||
       (isRPC && !isFWD && fabs(etaValue) > theGMTScales->getReducedEtaScale(1)->getScaleMax() )) {
    if(!m_saveFlag) edm::LogWarning("LUTRangeViolation") 
       << "L1MuGMTMIAUEtaConvLUT::TheLookupFunction: RPC " << (isFWD?"fwd":"brl") << " eta value out of range: " << etaValue;
  }
  else 
    eta4bit = theGMTScales->getReducedEtaScale(idx_drcr)->getPacked( etaValue );

  return eta4bit;
}














