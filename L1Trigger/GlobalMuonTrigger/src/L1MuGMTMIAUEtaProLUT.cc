//-------------------------------------------------
//
//   Class: L1MuGMTMIAUEtaProLUT
//
// 
//   $Date: 2008/04/17 23:18:30 $
//   $Revision: 1.7 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaProLUT.h"

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
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTEtaLUT.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTMIAUEtaProLUT::InitParameters() {
  m_IsolationCellSizeEta = L1MuGMTConfig::getIsolationCellSizeEta();
}

//--------------------------------------------------------------------------------
// Eta Projection LUT: Based on eta(6), pt(5) and charge(1), project in eta and directly 
// =================== output the eta select-bits for the regions in eta.
//  
// In the barrel there are 8 slect bits while in the forward chip there are 10.
//
// output: eta select bits
//                    eta= -3.                                     3. 
// total: 14 rings in eta   0  1  2  3  4  5  6  7  8  9 10 11 12 13
// 
// barrel:8                          0  1  2  3  4  5  6  7
// fwd:10                   0  1  2  3  4              5  6  7  8  9
//
// As for the phi projection, in case of MIP bits by defualt only one region is checked
// while for the ISO bit assignment multiple regions can be checked as given by 
// IsolationCellSizeEta. Adjacent regions to be ckecked are determined in analogy to
// the phi projection.
//
//--------------------------------------------------------------------------------

unsigned L1MuGMTMIAUEtaProLUT::TheLookupFunction (int idx, unsigned eta, unsigned pt, unsigned charge) const {
  // idx is MIP_DT, MIP_BRPC, ISO_DT, ISO_BRPC, MIP_CSC, MIP_FRPC, ISO_CSC, ISO_FRPC
  // INPUTS:  eta(6) pt(5) charge(1)
  // OUTPUTS: eta_sel(10) 

  // const L1MuGMTScales* theGMTScales = L1MuGMTConfig::getGMTScales();
  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();
  const L1MuTriggerPtScale* theTriggerPtScale = L1MuGMTConfig::getTriggerPtScale();
  const L1CaloGeometry* theCaloGeom = L1MuGMTConfig::getCaloGeom() ;

  int isRPC = idx % 2;
  int isFWD = idx / 4;
      
  int isISO = (idx / 2) % 2;

  int idx_drcr = isFWD * 2 + isRPC;

  if (pt==0) return 0; // empty candidate

  int ch_idx = (charge == 0) ? 1 : 0; // positive charge is 0 (but idx 1)

  float oldeta = theTriggerScales->getRegionalEtaScale(idx_drcr)->getCenter(eta);

  if (idx_drcr==2) oldeta = theTriggerScales->getRegionalEtaScale(idx_drcr)->getLowEdge(eta); //FIXME use center when changed in ORCA

  if ( (isRPC && isFWD && fabs(oldeta) < 1.04  ) ||
       (isRPC && !isFWD && fabs(oldeta) > 1.04 ) ) {
    if(!m_saveFlag) edm::LogWarning("LUTRangeViolation") 
                         << "L1MuGMTMIAUEtaProLUT::TheLookupFunction: RPC " << (isFWD?"fwd":"brl") 
	                 << " eta value out of range: " << oldeta;
  }

  // eta conversion depends only on isys by default
  int isys = isFWD + 2 * isRPC; // DT, CSC, BRPC, FRPC
  float neweta  =  L1MuGMTEtaLUT::eta (isys, isISO, ch_idx, oldeta, 
     theTriggerPtScale->getPtScale()->getLowEdge(pt) );  // use old LUT, here
  // theTriggerScales->getPtScale()->getLowEdge(pt) );  // use old LUT, here


  //  unsigned icenter = theGMTScales->getCaloEtaScale()->getPacked( neweta );
  // globalEtaIndex is 0-21 for forward+central; need to shift to 0-13 for central only
  unsigned icenter = theCaloGeom->globalEtaIndex( neweta ) - theCaloGeom->numberGctForwardEtaBinsPerHalf() ;

  unsigned eta_select_word_14 = 1 << icenter; // for the whole detector
    
  // for ISOlation bit assignment, multiple regions can be selected according to the IsolationCellSize
  if (isISO) {
    int imin = icenter - ( m_IsolationCellSizeEta-1 ) / 2;
    int imax = icenter + ( m_IsolationCellSizeEta-1 ) / 2;

    // for even number of isolation cells check the fine grain info
    if (m_IsolationCellSizeEta%2 == 0) {
      // float bincenter = theGMTScales->getCaloEtaScale()->getCenter( icenter );
      // globalEtaIndex is 0-21 for forward+central; need to shift to 0-13 for central only
      float bincenter = theCaloGeom->globalEtaBinCenter( icenter + theCaloGeom->numberGctForwardEtaBinsPerHalf() );
      if ( neweta > bincenter ) imax++;
      else imin--;
    }
    if (imin<0) imin=0; 
    if (imax>13) imax=13;

    for (int i=imin; i<=imax; i++ )
      eta_select_word_14 |= 1 << i ;
  }
    
  // generate select words for barrel (10 central bits)
  // and for forward (5+5 fwd bits) case
  unsigned eta_select_word;
  if (isFWD) {
    unsigned mask5 = (1<<5)-1;
    eta_select_word = eta_select_word_14 & mask5;
    eta_select_word |= ( eta_select_word_14 & (mask5 << 9) ) >> 4;
  }
  else {
    unsigned mask10 = (1<<10)-1;
    eta_select_word = ( eta_select_word_14 & (mask10 << 2) ) >> 2;
  }

  return eta_select_word;
}


















