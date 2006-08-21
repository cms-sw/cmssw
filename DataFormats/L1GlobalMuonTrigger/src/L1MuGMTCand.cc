//-------------------------------------------------
//
//   Class: L1MuGMTCand
//
//   Description: L1 Global Muon Trigger Candidate
//
//
//   $Date: 2006/07/03 15:18:05 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"
#include "SimG4Core/Notification/interface/Singleton.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//---------------------------------
//       class L1MuGMTCand
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuGMTCand::L1MuGMTCand() : m_name("L1MuGMTCand"), m_bx(0), m_dataWord(0) {

}


L1MuGMTCand::L1MuGMTCand(const L1MuGMTCand& mu) :
  m_name(mu.m_name), m_bx(mu.m_bx), m_dataWord(mu.m_dataWord){
}

L1MuGMTCand::L1MuGMTCand(unsigned data, int bx) : 
  m_name("L1MuGMTCand"), m_bx(bx) , m_dataWord(data){

}

//--------------
// Destructor --
//--------------
L1MuGMTCand::~L1MuGMTCand() {

  reset();

}


//--------------
// Operations --
//--------------

//
// reset Muon Track Candidate
//
void L1MuGMTCand::reset() {

  m_bx       = 0;
  m_dataWord = 0;

}

//
// return phi of track candidate in radians (bin low edge)
//
float L1MuGMTCand::phiValue() const {

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getPhiScale()->getLowEdge( phiIndex() );

}


//
// return eta of track candidate (bin center)
//
float L1MuGMTCand::etaValue() const {

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getGMTEtaScale()->getCenter( etaIndex() );

}


//
// return pt-value of track candidate in GeV
//
float L1MuGMTCand::ptValue() const {

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getPtScale()->getLowEdge( ptIndex() );

}


//
// convert pT to trigger Scale
//
unsigned int L1MuGMTCand::triggerScale(float value) const {

  L1MuTriggerScales* theTriggerScales = Singleton<L1MuTriggerScales>::instance();
  return theTriggerScales->getPtScale()->getPacked( value );

}


//
// Equal operator
//
bool L1MuGMTCand::operator==(const L1MuGMTCand& cand) const {

  if ( m_bx        != cand.m_bx )        return false; 
  if ( m_dataWord  != cand.m_dataWord )  return false;
  return true;

}


//
// Unequal operator
//
bool L1MuGMTCand::operator!=(const L1MuGMTCand& cand) const {

  if ( m_bx        != cand.m_bx )        return true;  
  if ( m_dataWord  != cand.m_dataWord )  return true;
  return false;

}


//
// print parameters of track candidate
//
void L1MuGMTCand::print() const {

  if ( !empty() ) {
    edm::LogVerbatim("GMT_Candidate_info")
         << setiosflags(ios::right | ios::adjustfield | ios::showpoint | ios::fixed)
         << "bx = " << setw(2) << bx() << " " << endl
         << "pt = "  << setw(5) << setprecision(1) << ptValue() << " GeV  "
         << "charge = " << setw(2) << charge() << "  "
         << "eta = " << setw(5) << setprecision(2) << etaValue() << "  "
         << "phi = " << setw(5) << setprecision(3) << phiValue() << " rad  "
         << "quality = " << setw(1) << quality() << "  "
         << "isolated = " << setw(1) << isol() << "  "
         << "mip = " << setw(1) << mip() << endl;
  }

}


//
// output stream operator for track candidate
//
ostream& operator<<(ostream& s, const L1MuGMTCand& id) {

  if ( !id.empty() ) {
    s << setiosflags(ios::showpoint | ios::fixed) 
      << "bx = " << setw(2) << id.bx() << "  "
      << "pt = "  << setw(5) << setprecision(1) << id.ptValue() << " GeV  "
      << "charge = " << setw(2) << id.charge() << "  " 
      << "eta = " << setw(5) << setprecision(2) << id.etaValue() << "  " 
      << "phi = " << setw(5) << setprecision(3) << id.phiValue() << " rad  "
      << "quality = " << setw(1) << id.quality() << "  "
      << "isolated = " << setw(1) << id.isol() << "  "
      << "mip = " << setw(1) << id.mip() << "  ";
  }
  return s;

}








