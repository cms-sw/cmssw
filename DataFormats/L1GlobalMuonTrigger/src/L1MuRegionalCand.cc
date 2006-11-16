//-------------------------------------------------
//
/** \class L1MuRegionalCand
 *  A regional muon trigger candidate as received by the GMT
*/
//
//   $Date: 2006/08/21 14:26:08 $
//   $Revision: 1.2 $
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

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuPacking.h"
#include "SimG4Core/Notification/interface/Singleton.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

/// get phi-value of muon candidate in radians (low edge of bin)
float L1MuRegionalCand::phiValue() const {

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getPhiScale()->getLowEdge( phi_packed() );
  
}
    
/// get eta-value of muon candidate
float L1MuRegionalCand::etaValue() const {
  
  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getRegionalEtaScale( type_idx() )->getCenter( eta_packed() );

}
    
/// get pt-value of muon candidate in GeV
float L1MuRegionalCand::ptValue() const {
      
  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getPtScale()->getLowEdge( pt_packed() );
  
}


/// Set Phi Value
void L1MuRegionalCand::setPhiValue(float phiVal) { 

  float eps = 1.e-5; // add an epsilon so that setting works with low edge value

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  unsigned phi = theTriggerScales->getPhiScale()->getPacked( phiVal + eps );

  writeDataField (PHI_START, PHI_LENGTH, phi); 
}

/// Set Pt Value
void L1MuRegionalCand::setPtValue(float ptVal) { 

  float eps = 1.e-5; // add an epsilon so that setting works with low edge value

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  unsigned pt = theTriggerScales->getPtScale()->getPacked( ptVal + eps );

  writeDataField (PT_START, PT_LENGTH, pt); 
}

/// Set Eta Value (need to set type, first)
void L1MuRegionalCand::setEtaValue(float etaVal) { 

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  unsigned eta = theTriggerScales->getRegionalEtaScale(type_idx())->getPacked( etaVal );

  writeDataField (ETA_START, ETA_LENGTH, eta); 
}

void L1MuRegionalCand::print() const {
  if ( !empty() ) {
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
	 << "type_idx = " << setw(1) << type_idx() << endl;
  }
}


unsigned L1MuRegionalCand::readDataField(unsigned start, unsigned count) const {
  unsigned mask = ( (1 << count) - 1 ) << start;
  return (m_dataWord & mask) >> start;
}

void L1MuRegionalCand::writeDataField(unsigned start, unsigned count, unsigned value) {
  if ( value < (unsigned)0 || value >= (unsigned)( 1 << count ) ) edm::LogWarning("ValueOutOfRange") 
         << "L1MuRegionalCand::writeDataField(): value " << value  
	 << " out of range for data field with bit width "  << count << endl << endl << endl;

  unsigned mask = ( (1 << count) - 1 ) << start;
  m_dataWord &= ~mask; // clear
  m_dataWord |= (value << start) & mask ;
}
