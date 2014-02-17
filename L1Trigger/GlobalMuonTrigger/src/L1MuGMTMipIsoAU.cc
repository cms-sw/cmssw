//-------------------------------------------------
//
//   Class: L1MuGMTMipIsoAU
//
//   Description:  GMT MIP & ISO bit assignment unit
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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMipIsoAU.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPSB.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPhiProjectionUnit.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTEtaProjectionUnit.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTMipIsoAU
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTMipIsoAU::L1MuGMTMipIsoAU(const L1MuGlobalMuonTrigger& gmt, int id) : 
  m_gmt(gmt), m_id(id), m_muons(8), m_MIP(8), m_ISO(8),
  m_MIP_PPUs(8), m_MIP_EPUs(8), m_ISO_PPUs(8), m_ISO_EPUs(8) {

  m_muons.reserve(8);
  m_MIP.reserve(8);
  m_ISO.reserve(8);
	
  // reserve MIP and ISO phi and eta projection units
  m_MIP_PPUs.reserve(8);
  m_MIP_EPUs.reserve(8);
  m_ISO_PPUs.reserve(8);
  m_ISO_EPUs.reserve(8);
	
  for (int i=0; i<8; i++) {
    m_MIP_PPUs[i]=new L1MuGMTPhiProjectionUnit (*this, 16 * m_id + i) ;
    m_MIP_EPUs[i]=new L1MuGMTEtaProjectionUnit (*this, 16 * m_id + i) ;
		
    m_ISO_PPUs[i]=new L1MuGMTPhiProjectionUnit (*this, 16 * m_id + 8 + i) ;
    m_ISO_EPUs[i]=new L1MuGMTEtaProjectionUnit (*this, 16 * m_id + 8 + i) ;
  }

}

//--------------
// Destructor --
//--------------
L1MuGMTMipIsoAU::~L1MuGMTMipIsoAU() { 

  reset();
	
  // delete MIP phi projection units	
  std::vector<L1MuGMTPhiProjectionUnit*>::iterator p_iter;
  for ( p_iter = m_MIP_PPUs.begin(); p_iter != m_MIP_PPUs.end(); p_iter++ )
    delete *p_iter;
  m_MIP_PPUs.clear();
	
  // delete ISO phi projection units	
  for ( p_iter = m_ISO_PPUs.begin(); p_iter != m_ISO_PPUs.end(); p_iter++ )
    delete *p_iter;
  m_ISO_PPUs.clear();

  // delete MIP eta projection units	
  std::vector<L1MuGMTEtaProjectionUnit*>::iterator e_iter;
  for ( e_iter = m_MIP_EPUs.begin(); e_iter != m_MIP_EPUs.end(); e_iter++ )
  	delete *e_iter;
  m_MIP_EPUs.clear();
	
  // delete ISO eta projection units	
  for ( e_iter = m_ISO_EPUs.begin(); e_iter != m_ISO_EPUs.end(); e_iter++ )
  	delete *e_iter;
  m_ISO_EPUs.clear();
}

//--------------
// Operations --
//--------------

//
// run MIP & ISO assignment unit
//
void L1MuGMTMipIsoAU::run() {

  load();

  // run MIP phi projection units	
  std::vector<L1MuGMTPhiProjectionUnit*>::iterator p_iter;
  for ( p_iter = m_MIP_PPUs.begin(); p_iter != m_MIP_PPUs.end(); p_iter++ )
    (*p_iter)->run();
	
  // run ISO phi projection units	
  for ( p_iter = m_ISO_PPUs.begin(); p_iter != m_ISO_PPUs.end(); p_iter++ )
    (*p_iter)->run();

  // run MIP eta projection units	
  std::vector<L1MuGMTEtaProjectionUnit*>::iterator e_iter;
  for ( e_iter = m_MIP_EPUs.begin(); e_iter != m_MIP_EPUs.end(); e_iter++ )
    (*e_iter)->run();
	
  // run ISO eta projection units	
  for ( e_iter = m_ISO_EPUs.begin(); e_iter != m_ISO_EPUs.end(); e_iter++ )
    (*e_iter)->run();
	
  assignMIP();
  assignISO();
}


//
// reset MIP & ISO assignment unit
//
void L1MuGMTMipIsoAU::reset() {

  for ( int i = 0; i < 8; i++ ) {  
    m_muons[i] = 0;
    m_MIP[i] = false;
    m_ISO[i] = false;
  }
	
  // reset MIP phi projection units	
  std::vector<L1MuGMTPhiProjectionUnit*>::iterator p_iter;
  for ( p_iter = m_MIP_PPUs.begin(); p_iter != m_MIP_PPUs.end(); p_iter++ ) {
    (*p_iter)->reset();
  }
	
  // reset ISO phi projection units	
  for ( p_iter = m_ISO_PPUs.begin(); p_iter != m_ISO_PPUs.end(); p_iter++ )
    (*p_iter)->reset();

  // reset MIP eta projection units	
  std::vector<L1MuGMTEtaProjectionUnit*>::iterator e_iter;
  for ( e_iter = m_MIP_EPUs.begin(); e_iter != m_MIP_EPUs.end(); e_iter++ )
    (*e_iter)->reset();
	
  // reset ISO eta projection units	
  for ( e_iter = m_ISO_EPUs.begin(); e_iter != m_ISO_EPUs.end(); e_iter++ )
    (*e_iter)->reset();
}


//
// print results of MIP & ISO assignment
//
void L1MuGMTMipIsoAU::print() const {

  std::stringstream outmip;
  outmip << "Assigned MIP bits : ";
  std::vector<bool>::const_iterator iter;
  for ( iter = m_MIP.begin(); iter != m_MIP.end(); iter++ ) {
    outmip << (*iter) << "  ";
  }
  edm::LogVerbatim("GMT_MipIso_info") << outmip.str();

  std::stringstream outiso;
  outiso << "Assigned ISO bits : ";
  for ( iter = m_ISO.begin(); iter != m_ISO.end(); iter++ ) {
    outiso << (*iter) << "  ";
  }
  edm::LogVerbatim("GMT_MipIso_info") << outiso.str();
}


//
// load MIP & ISO assignment unit (get data from PSB)
//
void L1MuGMTMipIsoAU::load() {

  // barrel MIP & ISO assignment unit gets DTBX and barrel RPC muons
  if ( m_id == 0 ) {
    for ( unsigned idt = 0; idt < L1MuGMTConfig::MAXDTBX; idt++ ) {
      m_muons[idt] = m_gmt.Data()->DTBXMuon(idt);
    }
    for ( unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCbarrel; irpc++ ) {
      m_muons[irpc+4] = m_gmt.Data()->RPCMuon(irpc);
    }
  }
  
  // endcap MIP & ISO assignment unit gets CSC and endcap RPC muons
  if ( m_id == 1 ) {  
    for ( unsigned icsc = 0; icsc < L1MuGMTConfig::MAXCSC; icsc++ ) {
      m_muons[icsc] = m_gmt.Data()->CSCMuon(icsc);
    }
    for ( unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCendcap; irpc++ ) {
      m_muons[irpc+4] = m_gmt.Data()->RPCMuon(irpc+4);
    }
  }
 
}

//
// run MIP assignment
//
void L1MuGMTMipIsoAU::assignMIP() {

  // get MIP bits from PSB
  const L1MuGMTMatrix<bool>& mip = m_gmt.Data()->mipBits();

  for ( int imuon = 0; imuon < 8; imuon++ ) 
    if (m_muons[imuon] && !m_muons[imuon]->empty() ) {
      bool tmpMIP=false;

      for ( int iphi = 0; iphi < 18; iphi++ )
	for ( int ieta = 0; ieta < 14; ieta++ ) {
	  if (m_MIP_PPUs[imuon]->isSelected(iphi) &&
	      m_MIP_EPUs[imuon]->isSelected(ieta) ) {
	    tmpMIP |= mip(ieta, iphi);
	    if ( L1MuGMTConfig::Debug(3) ) edm::LogVerbatim("GMT_MipIso_info") << "L1MuGMTMipIsoAU::assignMIP() checking calo region phi=" << 
	      iphi << ", eta="  << ieta;
	  }
	}
      m_MIP[imuon] = tmpMIP;
      m_gmt.DebugBlockForFill()->SetIsMIPISO( m_MIP_PPUs[imuon]->id(), tmpMIP?1:0) ;      
  }
}


//
// run ISO assignment
//
void L1MuGMTMipIsoAU::assignISO() {
  
  // get isolation bits from PSB
  const L1MuGMTMatrix<bool>& isol = m_gmt.Data()->isolBits();
  
  for ( int imuon = 0; imuon < 8; imuon++ )
    if (m_muons[imuon] && !m_muons[imuon]->empty() ) {
      bool tmpISO=true;
      bool any=false;

      for ( int iphi = 0; iphi < 18; iphi++ )
	for ( int ieta = 0; ieta < 14; ieta++ ) {
	  if (m_ISO_PPUs[imuon]->isSelected(iphi) &&
	      m_ISO_EPUs[imuon]->isSelected(ieta) ) {
	    tmpISO &= isol(ieta, iphi);
	    any  = true;
	    if ( L1MuGMTConfig::Debug(3) ) edm::LogVerbatim("GMT_MipIso_info") << "L1MuGMTMipIsoAU::assignISO() checking calo region phi=" << 
	      iphi << ", eta="  << ieta;
	  }
	}
      if (any) m_ISO[imuon] = tmpISO;
      else edm::LogWarning("MipISOProblem") << "L1MuGMTMipIsoAU::assignISO(): no calo region was checked!!"; 

      m_gmt.DebugBlockForFill()->SetIsMIPISO( m_ISO_PPUs[imuon]->id(), m_ISO[imuon]?1:0) ;      
    }
 
}







