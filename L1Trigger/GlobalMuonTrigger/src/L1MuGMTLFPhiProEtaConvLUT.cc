//-------------------------------------------------
//
//   Class: L1MuGMTLFPhiProEtaConvLUT
//
// 
//   $Date: 2006/07/07 16:57:06 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProEtaConvLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTScales.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuPacking.h"
#include "SimG4Core/Notification/interface/Singleton.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFPhiProEtaConvLUT::InitParameters() {
  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  m_theGMTScales = Singleton<L1MuGMTScales>::instance();
};

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

  int isRPC = idx % 2;
  int isFWD = idx / 2;
  
  float etaValue = m_theTriggerScales->getRegionalEtaScale(idx)->getCenter( eta_in );

  unsigned eta4bit = 0;
  if ( (isRPC && isFWD && fabs(etaValue) < m_theGMTScales->getReducedEtaScale(3)->getScaleMin() ) ||
       (isRPC && !isFWD && fabs(etaValue) > m_theGMTScales->getReducedEtaScale(1)->getScaleMax() )) {
    if(!m_saveFlag) edm::LogWarning("LUTRangeViolation") 
     << "L1MuGMTMIAUEtaConvLUT::TheLookupFunction: RPC " << (isFWD?"fwd":"brl") << " eta value out of range: " << etaValue << endl;
  }
  else 
    eta4bit = m_theGMTScales->getReducedEtaScale(idx)->getPacked( etaValue );

  return eta4bit;
}; 



















