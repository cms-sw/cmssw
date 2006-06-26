//-------------------------------------------------
//
/** \class L1MuRegionalCand
 *  A regional muon trigger candidate as received by the GMT
*/
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
//
//   Author :
//   H. Sakulin                    HEPHY Vienna
//
//--------------------------------------------------
using namespace std;

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuRegionalCand.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1DTTrackFinder/interface/L1MuTriggerScales.h"
#include "DataFormats/L1DTTrackFinder/interface/L1VCandidate.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuPacking.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

/// get phi-value of muon candidate in radians (low edge of bin)
float L1MuRegionalCand::phiValue() const {

  L1MuTriggerScales* theTriggerScales = new L1MuTriggerScales;
  float t_Scale = theTriggerScales->getPhiScale()->getLowEdge( phi_packed() );
  delete theTriggerScales;

  return t_Scale;
};
    
/// get eta-value of muon candidate
float L1MuRegionalCand::etaValue() const {
  
  L1MuTriggerScales* theTriggerScales = new L1MuTriggerScales;
  float t_Scale = theTriggerScales->getRegionalEtaScale( type_idx() )->getCenter( eta_packed() );
  delete theTriggerScales;

  return t_Scale;
};
    
/// get pt-value of muon candidate in GeV
float L1MuRegionalCand::ptValue() const {
      
  L1MuTriggerScales* theTriggerScales = new L1MuTriggerScales;
  float t_Scale =  theTriggerScales->getPtScale()->getLowEdge( pt_packed() );
  delete theTriggerScales;

  return t_Scale;
};


/// Set Phi Value
void L1MuRegionalCand::setPhiValue(float phiVal) { 

  float eps = 1.e-5; // add an epsilon so that setting works with low edge value

  L1MuTriggerScales* theTriggerScales = new L1MuTriggerScales;
  unsigned int phi = theTriggerScales->getPhiScale()->getPacked( phiVal + eps );
  delete theTriggerScales;

  writeDataField (PHI_START, PHI_LENGTH, phi); 
};

/// Set Pt Value
void L1MuRegionalCand::setPtValue(float ptVal) { 

  float eps = 1.e-5; // add an epsilon so that setting works with low edge value

  L1MuTriggerScales* theTriggerScales = new L1MuTriggerScales;
  unsigned int pt =theTriggerScales->getPtScale()->getPacked( ptVal + eps );
  delete theTriggerScales;

  writeDataField (PT_START, PT_LENGTH, pt); 
};

/// Set Eta Value (need to set type, first)
void L1MuRegionalCand::setEtaValue(float etaVal) { 

  L1MuTriggerScales* theTriggerScales = new L1MuTriggerScales;
  unsigned int eta = theTriggerScales->getRegionalEtaScale(type_idx())->getPacked( etaVal );
  delete theTriggerScales;

  writeDataField (ETA_START, ETA_LENGTH, eta); 
}; 

/// Set: to be removed when DT, CSC and RPC use L1MuRegionalCand
void L1MuRegionalCand::set(const L1VCandidate* muon, unsigned type, int bx) {

  m_dataWord = 0;      // clear
  m_bx = bx;

  if (type==0 || type==2) {
    cout << endl << endl
	 << "Error in L1MuRegionalCand::set(). Converting DT and CSC muons is no longer supported." 
	 << endl << endl;
    return;
  }

  if (! muon) { return;}

  // continue to convert RPC muons (to be fixed when RPC Trigger implements L1MuRegionalCand)

  writeDataField( PHI_START, PHI_LENGTH, muon->phi() );
  writeDataField( PT_START, PT_LENGTH, muon->pt() );
  writeDataField( QUAL_START, QUAL_LENGTH, muon->quality() );
  writeDataField( CHARGE_START, CHARGE_LENGTH, muon->charge()==1? 0 : 1 ); 
  writeDataField( CHVALID_START, CHVALID_LENGTH, 1 );  // FIXME when available

  unsigned eta = 0;
  L1MuSignedPacking<6> RPCEtaPacking;
  eta = RPCEtaPacking.packedFromIdx( ( (int) muon->eta()) - 32 ); //FIXME RPC eta coding

  writeDataField( ETA_START, ETA_LENGTH, eta);
  writeDataField (FINEHALO_START, FINEHALO_LENGTH, 0);
  writeDataField (TYPE_START, TYPE_LENGTH, type);
}


void L1MuRegionalCand::print() const {
  if ( !empty() ) { 
    cout.setf(ios::showpoint);
    cout.setf(ios::right,ios::adjustfield);
    cout << setiosflags(ios::showpoint | ios::fixed | ios::right | ios::adjustfield)
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
};


unsigned L1MuRegionalCand::readDataField(unsigned start, unsigned count) const {
  unsigned mask = ( (1 << count) - 1 ) << start;
  return (m_dataWord & mask) >> start;
}

void L1MuRegionalCand::writeDataField(unsigned start, unsigned count, unsigned value) {
  if ( value < 0 || value >= (unsigned)( 1 << count ) )
    cout << "*** error in L1MuRegionalCand::writeDataField(): value " << value  
	 << " out of range for data field with bit width "  << count << endl << endl << endl;

  unsigned mask = ( (1 << count) - 1 ) << start;
  m_dataWord &= ~mask; // clear
  m_dataWord |= (value << start) & mask ;
}
