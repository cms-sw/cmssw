//-------------------------------------------------
//
//   Class: L1MuGMTPhiProjectionUnit
//
//   Description: GMT Phi Projection Unit
//
//
//   $Date: 2007/04/10 09:59:19 $
//   $Revision: 1.3 $
//
//   Author :
//   H. Sakulin                CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPhiProjectionUnit.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMipIsoAU.h"

#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUPhiPro1LUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUPhiPro2LUT.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTPhiProjectionUnit
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTPhiProjectionUnit::L1MuGMTPhiProjectionUnit(const L1MuGMTMipIsoAU& miau, int id) : 
  m_MIAU(miau), m_id(id), m_mu(0) {
}

//--------------
// Destructor --
//--------------
L1MuGMTPhiProjectionUnit::~L1MuGMTPhiProjectionUnit() { 

  reset();
  
}

//--------------
// Operations --
//--------------
 
//
// run phi projection unit
//
void L1MuGMTPhiProjectionUnit::run() {

  load();
  if (m_mu && ( !m_mu->empty() ) ) {
    int lut_id = m_id / 4;

    // obtain inputs as coded in HW
    unsigned pt  = m_mu->pt_packed();
    unsigned charge  = m_mu->charge_packed();
    unsigned eta = m_mu->eta_packed();
    unsigned phi = m_mu->phi_packed();

    // split phi in fine and coarse
    unsigned phi_fine = phi & ( (1<<3) - 1 ); // lower 3 bits
    unsigned phi_coarse = phi >> 3; // upper 5 bits

    // eta conversion lookup
    L1MuGMTMIAUEtaConvLUT* etaconv_lut = L1MuGMTConfig::getMIAUEtaConvLUT();
    unsigned eta4bit = etaconv_lut->SpecificLookup_eta_out (lut_id, eta);
    
    // phi projection 1 lookups
    L1MuGMTMIAUPhiPro1LUT* phipro1_lut = L1MuGMTConfig::getMIAUPhiPro1LUT();
    unsigned cphi_fine = phipro1_lut->SpecificLookup_cphi_fine (lut_id, phi_fine, eta4bit, pt, charge);
    unsigned cphi_ofs  = phipro1_lut->SpecificLookup_cphi_ofs  (lut_id, phi_fine, eta4bit, pt, charge);

    // phi projection 2 lookup
    L1MuGMTMIAUPhiPro2LUT* phipro2_lut = L1MuGMTConfig::getMIAUPhiPro2LUT();
    unsigned phi_sel_bits = phipro2_lut->SpecificLookup_phi_sel (lut_id, phi_coarse, cphi_fine, cphi_ofs, charge);

    // convert to bit array
    //
    // see comments in L1MuGMTMIAUEtaProLUT.cc
    //
    m_phi_select = (unsigned) 0;
    
    //  shift by 9 bits  //FIXME: fix when calo delivers the MIP bits correctly!
    for (unsigned int i=0; i<9; i++)
      if ( (phi_sel_bits & (1 << i))  == (unsigned) (1<<i))
	m_phi_select[i+9] = 1;
    
    for (unsigned int i=9; i<18; i++)
      if ( (phi_sel_bits & (1 << i))  == (unsigned) (1<<i))
	m_phi_select[i-9] = 1;
    
    m_MIAU.GMT().DebugBlockForFill()->SetPhiSelBits( m_id, m_phi_select.to_ulong()) ;
  }
}


//
// reset phi projection unit
//
void L1MuGMTPhiProjectionUnit::reset() {

  m_mu = 0;
  m_iphi = 0;
  m_fphi = 0.;
  m_phi_select = (unsigned int) 0;
}


//
// print results of phi projection
//
void L1MuGMTPhiProjectionUnit::print() const {

  edm::LogVerbatim("GMT_PhiProjection_info") << "Phi select bits: ";
  for ( int i=0; i<18; i++ ) {
    edm::LogVerbatim("GMT_PhiProjection_info") << m_phi_select[i] << "  ";
  }
  edm::LogVerbatim("GMT_PhiProjection_info") << " ";
}


//
// load 1 muon into phi projection unit
//
void L1MuGMTPhiProjectionUnit::load() {

  // retrieve muon from MIP & ISO bit assignment unit	
  m_mu = m_MIAU.muon( m_id % 8 );
}


