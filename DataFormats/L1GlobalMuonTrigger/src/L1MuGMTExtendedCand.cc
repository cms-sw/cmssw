//-------------------------------------------------
//
//   Class: L1MuGMTExtendedCand
//
//   Description: L1 Global Muon Trigger Candidate
//
//
//   $Date: 2007/04/02 15:44:10 $
//   $Revision: 1.3 $
//
//   Author :
//   H. Sakulin        HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//---------------------------------
//       class L1MuGMTExtendedCand
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuGMTExtendedCand::L1MuGMTExtendedCand() : L1MuGMTCand(), m_rank(0) {
}

L1MuGMTExtendedCand::L1MuGMTExtendedCand(const L1MuGMTExtendedCand& mu) :
  L1MuGMTCand(mu), m_rank(mu.m_rank){
}

L1MuGMTExtendedCand::L1MuGMTExtendedCand(unsigned data, unsigned rank, int bx) : L1MuGMTCand (data, bx), m_rank(rank) {

}

//--------------
// Destructor --
//--------------
L1MuGMTExtendedCand::~L1MuGMTExtendedCand() {

  reset();

}


//--------------
// Operations --
//--------------

//
// reset Muon Track Candidate
//
void L1MuGMTExtendedCand::reset() {

  L1MuGMTCand::reset();
  m_rank = 0;

}

//
// return detector index 
// 1 RPC, 2 DT, 3 DT/RPC, 4 CSC, 5 CSC/RPC
//
// supported for backward compatibility only
//

unsigned int L1MuGMTExtendedCand::detector() const {
  
  if (quality() == 7) // matched ?
    return isFwd() ? 5 : 3;
  else 
    return isRPC() ? 1 : ( isFwd()? 4 : 2); 

}


//
// Equal operator
//
bool L1MuGMTExtendedCand::operator==(const L1MuGMTExtendedCand& cand) const {

  if ( (L1MuGMTCand const&) *this != cand )   return false; 
  if ( m_rank                     != m_rank ) return false;
  return true;

}


//
// Unequal operator
//
bool L1MuGMTExtendedCand::operator!=(const L1MuGMTExtendedCand& cand) const {

  if ( (L1MuGMTCand const&) *this != cand ) return true;  
  if ( m_rank  != cand.m_rank )             return true;
  return false;

}


//
// print parameters of track candidate
//
void L1MuGMTExtendedCand::print() const {

  L1MuGMTCand::print();
  if ( !empty() ) {
    edm::LogVerbatim("GMT_Candidate_info")
         << setiosflags(ios::right | ios::adjustfield | ios::showpoint | ios::fixed)
         << "rank = " << setw(3) << rank() << "  " 
         << "idxdtcsc = " << setw(1) << getDTCSCIndex() << "  "
         << "idxrpc = " << setw(1) << getRPCIndex() << "  "
         << "isFwd = " << setw(1) << isFwd()  << "  "
         << "isRPC = " << setw(1) << isRPC() << endl;
   }

}


//
// output stream operator for track candidate
//
ostream& operator<<(ostream& s, const L1MuGMTExtendedCand& id) {

  if ( !id.empty() ) {
    s << ( (L1MuGMTCand const &) id ) <<
      setiosflags(ios::showpoint | ios::fixed) 
      << "rank = " << setw(3) << id.rank() << "  "
      << "idxdtcsc = " << setw(1) << id.getDTCSCIndex() << "  "
      << "idxrpc = " << setw(1) << id.getRPCIndex() << "  "
      << "isFwd = " << setw(1) << id.isFwd() << "  "
      << "isRPC = " << setw(1) << id.isRPC() ;
  }
  return s;

}










