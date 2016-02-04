//-------------------------------------------------
//
//   Class: L1MuGMTMIAUPhiPro2LUT
//
// 
//   $Date: 2007/04/02 15:45:39 $
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUPhiPro2LUT.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTMIAUPhiPro2LUT::InitParameters() {
  m_IsolationCellSizePhi = L1MuGMTConfig::getIsolationCellSizePhi();
}

//--------------------------------------------------------------------------------
// Phi Projection LUT 2:  Set phi-select-bits based on start calo region(5bit), offset(3bit),
// =====================  fine-grain offset(1bit) and charge(1 bit)
//
// The 18 phi-select bits select which calo regions have to be checked for MIP
// and Quiet information. For MIP by default only one region is checked while for
// the Quiet bits multiple regions can be ckecked based on IsolationCellSizePhi.
// If IsolationCellSizePhi is even, then the fine-grain offset decides in which 
// direction from the central region to check additional regions.
// 
//--------------------------------------------------------------------------------

unsigned L1MuGMTMIAUPhiPro2LUT::TheLookupFunction (int idx, unsigned cphi_start, unsigned cphi_fine, unsigned cphi_ofs, unsigned charge) const {
  // idx is MIP_DT, MIP_BRPC, ISO_DT, ISO_BRPC, MIP_CSC, MIP_FRPC, ISO_CSC, ISO_FRPC
  // INPUTS:  cphi_start(5) cphi_fine(1) cphi_ofs(3) charge(1)
  // OUTPUTS: phi_sel(18) 

  // this LUT generates the 18 phi-select bits for the 18 Calo regions
  if (cphi_start > 17) return 0;

  int isISO = (idx / 2) % 2;
    
  int offset = ( int(cphi_ofs) - 1 ) * ( (charge==0) ? 1 : -1 );
  
  int center_region = ( 18 + int(cphi_start) + offset ) % 18;
    
  // for MIP bit assignment, only one region is selected
  unsigned phi_select_word = 1 << center_region;

  // for ISOlation bit assignment, multiple regions can be selected according to the IsolationCellSize
  if (isISO) {
    int imin = center_region - ( m_IsolationCellSizePhi-1 ) / 2;
    int imax = center_region + ( m_IsolationCellSizePhi-1 ) / 2;

    // for even number of isolation cells check the fine grain info
    if (m_IsolationCellSizePhi%2 == 0) {
      if ( cphi_fine==1 ) imax++;
      else imin--;
    }

    for (int i=imin; i<=imax; i++ )
      phi_select_word |= 1 << ( (i+18) % 18 );
  }

  return phi_select_word;
}


















