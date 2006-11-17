//-------------------------------------------------
//
//   Class: L1MuGMTMIAUEtaProLUT
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaProLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTScales.h"
#include "SimG4Core/Notification/interface/Singleton.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTEtaLUT.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTMIAUEtaProLUT::InitParameters() {
  m_theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  m_theGMTScales = Singleton<L1MuGMTScales>::instance();
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

  int isRPC = idx % 2;
  int isFWD = idx / 4;
      
  int isISO = (idx / 2) % 2;

  int idx_drcr = isFWD * 2 + isRPC;

  if (pt==0) return 0; // empty candidate

  int ch_idx = (charge == 0) ? 1 : 0; // positive charge is 0 (but idx 1)

  float oldeta = m_theTriggerScales->getRegionalEtaScale(idx_drcr)->getCenter(eta);

  if (idx_drcr==2) oldeta = m_theTriggerScales->getRegionalEtaScale(idx_drcr)->getLowEdge(eta); //FIXME use center when changed in ORCA

  if ( (isRPC && isFWD && fabs(oldeta) < 1.04  ) ||
       (isRPC && !isFWD && fabs(oldeta) > 1.04 ) ) {
    if(!m_saveFlag) edm::LogWarning("LUTRangeViolation") 
                         << "L1MuGMTMIAUEtaProLUT::TheLookupFunction: RPC " << (isFWD?"fwd":"brl") 
	                 << " eta value out of range: " << oldeta << endl;
  }

  // eta conversion depends only on isys by default
  int isys = isFWD + 2 * isRPC; // DT, CSC, BRPC, FRPC
  float neweta  =  L1MuGMTEtaLUT::eta (isys, isISO, ch_idx, oldeta, 
				       m_theTriggerScales->getPtScale()->getLowEdge(pt) );  // use old LUT, here


  unsigned icenter = m_theGMTScales->getCaloEtaScale()->getPacked( neweta );

  unsigned eta_select_word_14 = 1 << icenter; // for the whole detector
    
  // for ISOlation bit assignment, multiple regions can be selected according to the IsolationCellSize
  if (isISO) {
    int imin = icenter - ( m_IsolationCellSizeEta-1 ) / 2;
    int imax = icenter + ( m_IsolationCellSizeEta-1 ) / 2;

    // for even number of isolation cells check the fine grain info
    if (m_IsolationCellSizeEta%2 == 0) {
      float bincenter = m_theGMTScales->getCaloEtaScale()->getCenter( icenter );
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


















