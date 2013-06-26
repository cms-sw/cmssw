//-------------------------------------------------
//
//   Class: L1MuGMTLFOvlEtaConvLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFOvlEtaConvLUT.h"

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

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFOvlEtaConvLUT::InitParameters() {
}


//--------------------------------------------------------------------------------
// Overlap eta conversion LUT
// 
// convert global eta to a 4-bit pseudo-signed eta in the overlap region to be used in
// the COU matching units
//
// instances:
// ----------
//   barrel chip: DT, bRPC
//   barrel chip : ovlCSC 
//
//   forward chip: CSC bRPC
//   forward chip: ovlDT
//
//--------------------------------------------------------------------------------

unsigned L1MuGMTLFOvlEtaConvLUT::TheLookupFunction (int idx, unsigned eta6) const {
  // idx is DT, CSC, bRPC, fRPC, ovlCSC, ovlDT
  // INPUTS:  eta6(6)
  // OUTPUTS: eta_ovl(4) 

  const L1MuGMTScales* theGMTScales = L1MuGMTConfig::getGMTScales();
  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();

  int idx_drcr = 0;

  switch (idx) {
  case DT     : idx_drcr = 0; break;
  case CSC    : idx_drcr = 2; break;
  case bRPC   : idx_drcr = 1; break;
  case fRPC   : idx_drcr = 3; break;
  case ovlCSC : idx_drcr = 2; break;
  case ovlDT  : idx_drcr = 0; break;
  }

  float etaValue = theTriggerScales->getRegionalEtaScale(idx_drcr)->getCenter( eta6 );

  unsigned eta4bit = 0;
  if (fabs(etaValue) <  theGMTScales->getOvlEtaScale(idx_drcr)->getScaleMin() || 
      fabs(etaValue) >  theGMTScales->getOvlEtaScale(idx_drcr)->getScaleMax() ) {
    eta4bit = 7; // out of range code is max pos value
  }

  else {
    eta4bit  = theGMTScales->getOvlEtaScale(idx_drcr)->getPacked( etaValue );
    //    cout << "etaValue  = " << etaValue << "   eta OVERLAP= " << eta4bit << endl;
  }

  return eta4bit;
}












