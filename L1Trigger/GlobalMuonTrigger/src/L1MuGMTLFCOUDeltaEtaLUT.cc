//-------------------------------------------------
//
//   Class: L1MuGMTLFCOUDeltaEtaLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFCOUDeltaEtaLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFCOUDeltaEtaLUT::InitParameters() {
}

//---------------------------------------------------------------------------
// Cancel-Out-Unit Delta-Eta LUT
// =============================
// 
// Calculate delta-eta between two eta values coded in a special 4-bit scale 
// used in the overlap region
//
// This scale is used to save I/O pins and in order to be able to do the delta-eta 
// calculateion in a distributed RAM LUT
//
//---------------------------------------------------------------------------

unsigned L1MuGMTLFCOUDeltaEtaLUT::TheLookupFunction (int idx, unsigned eta1, unsigned eta2) const {
  // idx is DTCSC, CSCDT, bRPCCSC, fRPCDT
  // INPUTS:  eta1(4) eta2(4)
  // OUTPUTS: delta_eta(4) 

  const L1MuGMTScales* theGMTScales = L1MuGMTConfig::getGMTScales();

  // check out of range in inputs
  L1MuSignedPacking<4> pack;
  unsigned delta_eta_OOR = pack.packedFromIdx (-8);

  if (eta1==7 || eta2 ==7) return delta_eta_OOR;

  int type1=0, type2=0;
  switch (idx) {
    case 0: type1 = 0; type2 = 2; break; // DT, CSC
    case 1: type1 = 2; type2 = 0; break; // CSC, DT
    case 2: type1 = 1; type2 = 2; break; // bRPC, CSC
    case 3: type1 = 3; type2 = 0; break; // fRPC, DT
  }

  float etaValue1 = theGMTScales->getOvlEtaScale(type1)->getCenter(eta1);
  float etaValue2 = theGMTScales->getOvlEtaScale(type2)->getCenter(eta2); 

  float delta_eta = etaValue1 - etaValue2;

  unsigned delta_eta_4bit = 0;

  // check out of range
  if (delta_eta < theGMTScales->getDeltaEtaScale(idx+2)->getScaleMin() ||
      delta_eta > theGMTScales->getDeltaEtaScale(idx+2)->getScaleMax()) {
    delta_eta_4bit = delta_eta_OOR;
  }
  else {
    delta_eta_4bit = theGMTScales->getDeltaEtaScale(idx+2)->getPacked( delta_eta );
  }
    
  return delta_eta_4bit;
}



















