//-------------------------------------------------
//
//   Class: L1MuGMTLFPhiProLUT
//
// 
//   $Date: 2008/04/21 17:22:41 $
//   $Revision: 1.5 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPhiLUT.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFPhiProLUT::InitParameters() {
}

//------------------------
// The Lookup Function  --
//------------------------

//
// Phi projection LUT to project the phi coordinates to the vertex
// 
// The output is 9 bits signed and should be in the range of -32 <= dphi < 32
// (+/- 80 deg)
// 
// The input eta is converted from six to four bits as in the MIP/ISO AU chips
//
// The same parameterization as in the MIP/ISO AU chips can be used (proj. to vertex for ISO).

unsigned L1MuGMTLFPhiProLUT::TheLookupFunction (int idx, unsigned eta, unsigned pt, unsigned charge) const {
  // idx is DT, BRPC, CSC, FRPC
  // INPUTS:  eta(4) pt(5) charge(1)
  // OUTPUTS: dphi(9) 

//  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();
  const L1MuTriggerPtScale* theTriggerPtScale = L1MuGMTConfig::getTriggerPtScale();

  //  static bool doProjection = SimpleConfigurable<bool> (false, "L1GlobalMuonTrigger:PropagatePhi" );
  static bool doProjection = L1MuGMTConfig::getPropagatePhi();

  if (!doProjection) return 0;

  int isRPC = idx % 2;
  int isFWD = idx / 2;
      
  int isys = isFWD + 2 * isRPC; // DT, CSC, BRPC, FRPC
  int ch_idx = (charge == 0) ? 1 : 0; // positive charge is 0 (but idx 1)

  // currently only support 3-bit eta (3 lower bits); ignore 4th bit
  if (eta>7) eta -= 8;

  float dphi =  L1MuGMTPhiLUT::dphi (isys, 1, ch_idx, (int) eta, 
     theTriggerPtScale->getPtScale()->getLowEdge(pt) );  // use old LUT, here
  // theTriggerScales->getPtScale()->getLowEdge(pt) );  // use old LUT, here
  
  int dphi_int = (int) ( (-dphi + 1.25 / 180. * M_PI + 2* M_PI ) / ( 2.5 / 180. * M_PI ) ) - 144;
    
  L1MuSignedPacking<9> PhiPacking;
  return PhiPacking.packedFromIdx(dphi_int);
}



















