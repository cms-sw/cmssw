//-------------------------------------------------
//
//   Class: L1MuGMTCancelOutUnit
//
//   Description: DT/CSC cancel-out unit
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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTCancelOutUnit.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatcher.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTReg.h"

// --------------------------------
//       class L1MuGMTCancelOutUnit
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTCancelOutUnit::L1MuGMTCancelOutUnit(const L1MuGlobalMuonTrigger& gmt, int id) : 
               m_gmt(gmt), m_id(id), m_matcher(gmt, id+2), m_MyChipCancelbits(4), m_OtherChipCancelbits(4) {

}

//--------------
// Destructor --
//--------------
L1MuGMTCancelOutUnit::~L1MuGMTCancelOutUnit() { 

  reset();
  
}

//--------------
// Operations --
//--------------

//
// run cancel-out unit
//
void L1MuGMTCancelOutUnit::run() {

  m_matcher.run();
  if (L1MuGMTConfig::Debug(3)) {
    edm::LogVerbatim("GMT_CancelOut_info") << "result of cancel-out matcher: ";
    m_matcher.print();
  }
  decide(); 
}

//
// clear cancel-out unit
//
void L1MuGMTCancelOutUnit::reset() {

  m_matcher.reset();
  
  for ( int i = 0; i < 4; i++ ) {
    m_MyChipCancelbits[i] = false;
    m_OtherChipCancelbits[i] = false;
  }  
}


//
// print cancel-out results
//
void L1MuGMTCancelOutUnit::print() {

  std::stringstream outmy;
  switch (m_id) {
    case 0 : outmy << "DT  " ; break;
    case 1 : outmy << "CSC " ; break;
    case 2 : outmy << "bRPC" ; break;
    case 3 : outmy << "fRPC" ; break;
  }
  outmy << "(my chip) cancel-bits : " ;
  for ( int i = 0; i < 4; i++ ) outmy << m_MyChipCancelbits[i] << "  ";
  edm::LogVerbatim("GMT_CancelOut_info") << outmy.str();

  std::stringstream outother;
  if (m_id==2 || m_id==3) {
    outother << (m_id==2 ? "CSC" : "DT" ) <<  "(other chip) cancel-bits : " ;
    for ( int i = 0; i < 4; i++ ) outother << m_OtherChipCancelbits[i] << "  ";
    outother << std::endl;
  }
  edm::LogVerbatim("GMT_CancelOut_info") << outother.str();
}



//
// compute cancel decision
//
void L1MuGMTCancelOutUnit::decide() {

  // CancelDecisionLogic configuration register
  //
  unsigned CDL_config = L1MuGMTConfig::getRegCDLConfig()->getValue(m_id);

  // compute cancel decsion for my chip muons (mine)

  for(int imine=0; imine<4; imine++) {
    int idxother = m_matcher.pairM().rowAny(imine);
    if (idxother != -1) {
      int mine_is_matched = 0;
      switch(m_id) {
      case 0: mine_is_matched = m_gmt.Matcher(0)->pairM().rowAny(imine) != -1; break; //DT
      case 1: mine_is_matched = m_gmt.Matcher(1)->pairM().rowAny(imine) != -1; break; //CSC
      case 2: mine_is_matched = m_gmt.Matcher(0)->pairM().colAny(imine) != -1; break; //bRPC
      case 3: mine_is_matched = m_gmt.Matcher(1)->pairM().colAny(imine) != -1; break; //fRPC
      }
      int other_is_matched = m_gmt.Matcher( 1-(m_id%2) )->pairM().rowAny(idxother) != -1;

      // calculate address of bit in CDL_config register
      unsigned addr = (unsigned) (2*mine_is_matched + other_is_matched);
      unsigned mask = (unsigned) 1 << addr;
      
      m_MyChipCancelbits[imine] = (CDL_config & mask) == mask;
    }
  }

  // compute cancel decsison for other chip muons (other)

  for(int iother=0; iother<4; iother++) {
    int idxmine = m_matcher.pairM().colAny(iother);
    if (idxmine != -1) {
      int mine_is_matched = 0;
      switch(m_id) {
      case 0: mine_is_matched = m_gmt.Matcher(0)->pairM().rowAny(idxmine) != -1; break; //DT
      case 1: mine_is_matched = m_gmt.Matcher(1)->pairM().rowAny(idxmine) != -1; break; //CSC
      case 2: mine_is_matched = m_gmt.Matcher(0)->pairM().colAny(idxmine) != -1; break; //bRPC
      case 3: mine_is_matched = m_gmt.Matcher(1)->pairM().colAny(idxmine) != -1; break; //fRPC
      }
      int other_is_matched = m_gmt.Matcher( 1-(m_id%2) )->pairM().rowAny(iother) != -1;

      // calculate address of bit in CDL_config register
      unsigned addr = (unsigned) (2*other_is_matched + mine_is_matched);
      unsigned mask = (unsigned)1 << (addr+4);
      
      m_OtherChipCancelbits[iother] = (CDL_config & mask) == mask;
    }
  }

  m_gmt.DebugBlockForFill()->SetCancelBits( m_id, m_MyChipCancelbits, m_OtherChipCancelbits) ; 
}

  











