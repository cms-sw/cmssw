//-------------------------------------------------
//
//   Class: L1MuGMTLFEtaConvLUT
//
// 
//   $Date: 2007/04/02 15:45:38 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFEtaConvLUT.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"


//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFEtaConvLUT::InitParameters() {
}

//----------------------------------------------------------------
// Main eta conversion LUT
//
// Converts 6 bit regional trigger eta to GMT output eta scale
//
//----------------------------------------------------------------

unsigned L1MuGMTLFEtaConvLUT::TheLookupFunction (int idx, unsigned eta_regional) const {
  // idx is DT, bRPC, CSC, fRPC
  // INPUTS:  eta_regional(6)
  // OUTPUTS: eta_gmt(6) 
 
  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();

  int isRPC = idx % 2;
  //  int isFWD = idx / 2;
  
  float etaValue = theTriggerScales->getRegionalEtaScale(idx)->getCenter( eta_regional );

  if ( fabs(etaValue) > 2.4 ) etaValue = 2.39 * ((etaValue)>0?1.:-1.);

  // this is the only trick needed ...
  if (isRPC) {
    // etaValue() is center. make eta=0 and |eta|=1.3 non-ambiguous, when converting to GMT bins
    etaValue += (etaValue>0.01? -1. : 1.) * 0.001;
  }
  
  unsigned eta_gmt = theTriggerScales->getGMTEtaScale()->getPacked( etaValue );

  return eta_gmt;
}



















