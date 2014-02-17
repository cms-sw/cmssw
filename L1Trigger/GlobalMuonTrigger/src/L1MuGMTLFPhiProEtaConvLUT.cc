//-------------------------------------------------
//
//   Class: L1MuGMTLFPhiProEtaConvLUT
//
// 
//   $Date: 2007/04/02 15:45:38 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProEtaConvLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFPhiProEtaConvLUT::InitParameters() {
}

//------------------------
// The Lookup Function  --
//------------------------
//
// The LUT converts eta from 6 to 4 bits in order to use it as an Input to the Phi Projection
// LUT in the Logic FPGA. It uses the same Scales as in the MIP/ISO AU Chip.
//

unsigned L1MuGMTLFPhiProEtaConvLUT::TheLookupFunction (int idx, unsigned eta_in) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta_in(6)
  // OUTPUTS: eta_out(4) 

  const L1MuGMTScales* theGMTScales = L1MuGMTConfig::getGMTScales();
  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();

  int isRPC = idx % 2;
  int isFWD = idx / 2;
  
  float etaValue = theTriggerScales->getRegionalEtaScale(idx)->getCenter( eta_in );

  unsigned eta4bit = 0;
  if ( (isRPC && isFWD && fabs(etaValue) < theGMTScales->getReducedEtaScale(3)->getScaleMin() ) ||
       (isRPC && !isFWD && fabs(etaValue) > theGMTScales->getReducedEtaScale(1)->getScaleMax() )) {
    if(!m_saveFlag) edm::LogWarning("LUTRangeViolation") 
     << "L1MuGMTMIAUEtaConvLUT::TheLookupFunction: RPC " << (isFWD?"fwd":"brl") << " eta value out of range: " << etaValue;
  }
  else 
    eta4bit = theGMTScales->getReducedEtaScale(idx)->getPacked( etaValue );

  return eta4bit;
}



















