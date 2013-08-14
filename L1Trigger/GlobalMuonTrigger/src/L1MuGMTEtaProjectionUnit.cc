//-------------------------------------------------
//
//   Class: L1MuGMTEtaProjectionUnit
//
//   Description: GMT Eta Projection Unit
//
//
//   $Date: 2007/04/10 09:59:19 $
//   $Revision: 1.4 $
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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTEtaProjectionUnit.h"
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
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMIAUEtaProLUT.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// --------------------------------
//       class L1MuGMTEtaProjectionUnit
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTEtaProjectionUnit::L1MuGMTEtaProjectionUnit(const L1MuGMTMipIsoAU& miau, int id) : 
  m_MIAU(miau), m_id(id), m_mu(0) {

}

//--------------
// Destructor --
//--------------
L1MuGMTEtaProjectionUnit::~L1MuGMTEtaProjectionUnit() { 

  reset();
  
}

//--------------
// Operations --
//--------------
 
//
// run eta projection unit
//
void L1MuGMTEtaProjectionUnit::run() {

  load();
  if ( m_mu && ( !m_mu->empty() ) ) {
    
    int isFwd = m_id / 16;
    int lut_id = m_id / 4;

    // obtain inputs as coded in HW
    unsigned pt  = m_mu->pt_packed();
    unsigned charge  = m_mu->charge_packed();
    unsigned eta = m_mu->eta_packed();
    
    // lookup
    L1MuGMTMIAUEtaProLUT* ep_lut = L1MuGMTConfig::getMIAUEtaProLUT();
    unsigned eta_sel_bits = ep_lut->SpecificLookup_eta_sel (lut_id, eta, pt, charge);

    // convert to bit array
    //
    // see comments in L1MuGMTMIAUEtaProLUT.cc
    //
    m_eta_select = (unsigned) 0;
    
    if (isFwd) { // forward
      for (int i=0; i<5; i++)
	if ( (eta_sel_bits & (1 << i))  == (unsigned) (1<<i))
	  m_eta_select[i] = 1;
      
      for (int i=5; i<10; i++)
	if ( (eta_sel_bits & (1 << i))  == (unsigned) (1<<i))
	  m_eta_select[i+4] = 1;            
    } else { // barrel
      for (int i=0; i<10; i++)
	if ( (eta_sel_bits & (1 << i))  == (unsigned) (1<<i))
	  m_eta_select[i+2] = 1;
    }
    
    //    m_MIAU.GMT().DebugBlockForFill()->SetEtaSelBits( m_id, m_eta_select.read(0,14)) ;
    m_MIAU.GMT().DebugBlockForFill()->SetEtaSelBits( m_id, m_eta_select.to_ulong()) ;
  }
}


//
// reset eta projection unit
//
void L1MuGMTEtaProjectionUnit::reset() {

  m_mu = 0;
  m_ieta = 0;
  m_feta = 0.;
  m_eta_select = (unsigned int) 0;
}


//
// print results of eta projection
//
void L1MuGMTEtaProjectionUnit::print() const {

  edm::LogVerbatim("GMT_EtaProjection_info") << "Eta select bits: ";
  for ( int i=0; i<14; i++ ) {
    edm::LogVerbatim("GMT_EtaProjection_info") << m_eta_select[i] << "  ";
  }
  edm::LogVerbatim("GMT_EtaProjection_info");
}


//
// load 1 muon into eta projection unit
//
void L1MuGMTEtaProjectionUnit::load() {

  // retrieve muon from MIP & ISO bit assignment unit	
  m_mu = m_MIAU.muon( m_id % 8 );
}














