//-------------------------------------------------
//
//   Class: L1MuGMTLFPtMixLUT
//
//
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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFPtMixLUT::InitParameters() {}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFPtMixLUT::TheLookupFunction(int idx, unsigned pt_dtcsc, unsigned pt_rpc) const {
  // idx is DTRPC, CSCRPC
  // INPUTS:  pt_dtcsc(5) pt_rpc(5)
  // OUTPUTS: pt_mixed(5)

  // choosing maximum pt

  return pt_dtcsc < pt_rpc ? pt_rpc : pt_dtcsc;
}
