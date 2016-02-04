//-------------------------------------------------
//
/** \class L1MuRegionalCand
 *  A regional muon trigger candidate as received by the GMT
*/
//
//   $Date: 2010/12/06 20:04:17 $
//   $Revision: 1.7 $
//
//   Author :
//   H. Sakulin                    HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//              ---------------------
//              -- Class Interface --
//              ---------------------

const float L1MuRegionalCand::m_invalidValue = -10.;

    /// constructor from data word
L1MuRegionalCand::L1MuRegionalCand(unsigned dataword, int bx) : m_bx(bx), m_dataWord(dataword) {
  m_phiValue = m_invalidValue; 
  m_etaValue = m_invalidValue;
  m_ptValue = m_invalidValue;
}

    /// constructor from packed members
L1MuRegionalCand::L1MuRegionalCand(unsigned type_idx, 
                     unsigned phi, unsigned eta, unsigned pt, unsigned charge,
		     unsigned ch_valid, unsigned finehalo, unsigned quality, int bx) : 
                     m_bx(bx), m_dataWord(0) {
  setType(type_idx);
  setPhiPacked(phi);
  setEtaPacked(eta);
  setPtPacked(pt);
  setChargePacked(charge);
  setChargeValidPacked(ch_valid);
  setFineHaloPacked(finehalo);
  setQualityPacked(quality);     
  m_phiValue = m_invalidValue; 
  m_etaValue = m_invalidValue;
  m_ptValue = m_invalidValue;
}

void L1MuRegionalCand::reset() {

  m_bx       = 0;
  m_dataWord = 0;
  m_phiValue = m_invalidValue; 
  m_etaValue = m_invalidValue;
  m_ptValue = m_invalidValue;

}

float L1MuRegionalCand::phiValue() const {
  if(m_phiValue == m_invalidValue) {
    edm::LogWarning("ValueInvalid") << 
     "L1MuRegionalCand::phiValue requested physical value is invalid";
  }
  return m_phiValue;
}

float L1MuRegionalCand::etaValue() const {
  if(m_etaValue == m_invalidValue) {
    edm::LogWarning("ValueInvalid") << 
     "L1MuRegionalCand::etaValue requested physical value is invalid";
  }
  return m_etaValue;
}

float L1MuRegionalCand::ptValue() const {
  if(m_ptValue == m_invalidValue) {
    edm::LogWarning("ValueInvalid") << 
     "L1MuRegionalCand::ptValue requested physical value is invalid";
  }
  return m_ptValue;
}

void L1MuRegionalCand::print() const {
  if ( !empty() ) {
    if(m_phiValue == m_invalidValue ||
       m_etaValue == m_invalidValue ||
       m_ptValue == m_invalidValue) {
      edm::LogVerbatim("GMT_Input_info")
           << setiosflags(ios::showpoint | ios::fixed | ios::right | ios::adjustfield)
       << "pt(index) = " << setw(2) << setprecision(1) << pt_packed() << "  "
       << "charge = " << setw(2) << chargeValue() << "  "
       << "eta(index) = " << setw(2) << eta_packed() << "  "
       << "phi(index) = " << setw(3) << phi_packed() << "  "
       << "quality = " << setw(1) << quality() << "  "
       << "charge_valid = " << setw(1) << chargeValid() << "  "
       << "fine_halo = " << setw(1) << isFineHalo() << "  "
       << "bx = " << setw(3) << bx() << "  " 
       << "type_idx = " << setw(1) << type_idx();
    } else {
      edm::LogVerbatim("GMT_Input_info")
           << setiosflags(ios::showpoint | ios::fixed | ios::right | ios::adjustfield)
	   << "pt = " << setw(5) << setprecision(1) << ptValue() << " GeV  "
	   << "charge = " << setw(2) << chargeValue() << " "
	   << "eta = " << setw(6) << setprecision(3) << etaValue() << "  "
	   << "phi = " << setw(5) << setprecision(3) << phiValue() << " rad  "
	   << "quality = " << setw(1) << quality() << "  "
	   << "charge_valid = " << setw(1) << chargeValid() << "  "
	   << "fine_halo = " << setw(1) << isFineHalo() << "  "
	   << "bx = " << setw(3) << bx() << "  " 
	   << "type_idx = " << setw(1) << type_idx();
    }
  }
}


unsigned L1MuRegionalCand::readDataField(unsigned start, unsigned count) const {
  unsigned mask = ( (1 << count) - 1 ) << start;
  return (m_dataWord & mask) >> start;
}

void L1MuRegionalCand::writeDataField(unsigned start, unsigned count, unsigned value) {
  if ( value >= ( 1U << count ) ) edm::LogWarning("ValueOutOfRange") // value >= 0, since value is unsigned
         << "L1MuRegionalCand::writeDataField(): value " << value  
	 << " out of range for data field with bit width "  << count;

  unsigned mask = ( (1 << count) - 1 ) << start;
  m_dataWord &= ~mask; // clear
  m_dataWord |= (value << start) & mask ;
}

