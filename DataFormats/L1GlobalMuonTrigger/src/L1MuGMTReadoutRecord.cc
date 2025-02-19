//-------------------------------------------------
//
/** \class L1MuGMTReadoutRecord
 *
 *  L1 Global Muon Trigger Readout Record
 *
 *  Contains the data that the GMT will send to readout
 *  for one bunch crossing.
*/
//
//   $Date: 2010/07/12 08:38:50 $
//   $Revision: 1.8 $
//
//   Author :
//   H. Sakulin                  HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
 
//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

//--------------------------------------
//       class L1MuGMTReadoutRecord
//--------------------------------------

//----------------
// Constructors --
//----------------

L1MuGMTReadoutRecord::L1MuGMTReadoutRecord() : m_InputCands(16), 
   m_BarrelCands(4), m_ForwardCands(4), m_GMTCands(4) {
  reset();
}
 
L1MuGMTReadoutRecord::L1MuGMTReadoutRecord(int bxie) : m_InputCands(16), 
   m_BarrelCands(4), m_ForwardCands(4), m_GMTCands(4) {
  reset();
  m_BxInEvent = bxie;
}
 
//--------------
// Destructor --
//--------------
L1MuGMTReadoutRecord::~L1MuGMTReadoutRecord() {
}

//--------------
// Operations --
//--------------

/// reset the record
void L1MuGMTReadoutRecord::reset() {

  m_BxNr = 0;
  m_BxInEvent = 0;
  m_EvNr = 0;
  m_BCERR = 0;

  std::vector<L1MuRegionalCand>::iterator itr;
  for(itr = m_InputCands.begin(); itr != m_InputCands.end(); itr++) (*itr).reset();

  std::vector<L1MuGMTExtendedCand>::iterator itg;
  for(itg = m_BarrelCands.begin(); itg != m_BarrelCands.end(); itg++) (*itg).reset();
  for(itg = m_ForwardCands.begin(); itg != m_ForwardCands.end(); itg++) (*itg).reset();
  for(itg = m_GMTCands.begin(); itg != m_GMTCands.end(); itg++) (*itg).reset();
  for(int i=0;i<8;i++) {
    m_MIPbits[i]=0;
    m_Quietbits[i]=0;
  }
}

/// get GMT candidates vector
vector<L1MuGMTExtendedCand>  L1MuGMTReadoutRecord::getGMTCands() const {

  vector<L1MuGMTExtendedCand> cands;

  std::vector<L1MuGMTExtendedCand>::const_iterator it;
  for(it = m_BarrelCands.begin(); it != m_BarrelCands.end(); it++) {
    if((*it).getDataWord()!=0) cands.push_back(*it);
  }
  for(it = m_ForwardCands.begin(); it != m_ForwardCands.end(); it++) {
    if((*it).getDataWord()!=0) cands.push_back(*it);
  }
    
  // sort by rank
  stable_sort( cands.begin(), cands.end(), L1MuGMTExtendedCand::RankRef() );


  return cands;
}

/// get GMT candidates vector as stored in data (no rank info)
vector<L1MuGMTExtendedCand>&  L1MuGMTReadoutRecord::getGMTCandsData() {

  return m_GMTCands;

}

/// get GMT barrel candidates vector
vector<L1MuGMTExtendedCand> L1MuGMTReadoutRecord::getGMTBrlCands() const {
  
  vector<L1MuGMTExtendedCand> cands;
  std::vector<L1MuGMTExtendedCand>::const_iterator it;
  for(it = m_BarrelCands.begin(); it != m_BarrelCands.end(); it++) {
    if((*it).getDataWord()!=0) cands.push_back(*it);
  }

  return cands;

}

/// get GMT forward candidates vector
vector<L1MuGMTExtendedCand> L1MuGMTReadoutRecord::getGMTFwdCands() const {

  vector<L1MuGMTExtendedCand> cands;
  std::vector<L1MuGMTExtendedCand>::const_iterator it;
  for(it = m_ForwardCands.begin(); it != m_ForwardCands.end(); it++) {
    if((*it).getDataWord()!=0) cands.push_back(*it);
  }

  return cands;

}

/// get DT candidates vector
vector<L1MuRegionalCand> L1MuGMTReadoutRecord::getDTBXCands() const {

  vector<L1MuRegionalCand> cands;
  
  for (int i=0; i<4; i++)
    if(m_InputCands[i].getDataWord() != 0)
      cands.push_back( m_InputCands[i] );
  

  return cands;
}


/// get CSC candidates vector
vector<L1MuRegionalCand> L1MuGMTReadoutRecord::getCSCCands() const {

  vector<L1MuRegionalCand> cands;
  
  for (int i=0; i<4; i++) 
    if(m_InputCands[i+8].getDataWord() != 0)
      cands.push_back( m_InputCands[i+8] );
  
  return cands;
}

/// get barrel RPC candidates vector
vector<L1MuRegionalCand> L1MuGMTReadoutRecord::getBrlRPCCands() const {

  vector<L1MuRegionalCand> cands;
  
  for (int i=0; i<4; i++) 
    if(m_InputCands[i+4].getDataWord() != 0)
      cands.push_back( m_InputCands[i+4] );
  
  return cands;
}

/// get forward RPC candidates vector
vector<L1MuRegionalCand> L1MuGMTReadoutRecord::getFwdRPCCands() const {

  vector<L1MuRegionalCand> cands;
  
  for (int i=0; i<4; i++) 
    if(m_InputCands[i+12].getDataWord() != 0)
      cands.push_back( m_InputCands[i+12] );
  
  return cands;
}

/// get MIP bit
unsigned L1MuGMTReadoutRecord::getMIPbit(int eta, int phi) const {

  if (phi<0 || phi > 17 || eta < 0 || eta > 13) return 0;

  int idx = eta * 18 + phi;
  int idx_word = idx / 32;
  int idx_bit = idx % 32;

  unsigned mask = 1 << (idx_bit-1);

  return( m_MIPbits[idx_word] & mask) ? 1 : 0;

}


/// get Quiet bit
unsigned L1MuGMTReadoutRecord::getQuietbit(int eta, int phi) const {

  if (phi<0 || phi > 17 || eta < 0 || eta > 13) return 0;

  int idx = eta * 18 + phi;
  int idx_word = idx / 32;
  int idx_bit = idx % 32;

  unsigned mask = 1 << (idx_bit-1);

  return( m_Quietbits[idx_word] & mask) ? 1 : 0;

}



//
// Setters
//

/// set Regional Candidates
void L1MuGMTReadoutRecord::setInputCand(int nr, L1MuRegionalCand const& cand) {
  if (nr>=0 && nr < 16) {
    m_InputCands[nr] = cand;
  }
}

/// set Regional Candidates
void L1MuGMTReadoutRecord::setInputCand(int nr, unsigned data) {
  if (nr>=0 && nr < 16) {
    m_InputCands[nr] = L1MuRegionalCand(data,m_BxInEvent); 
  }
}

/// set GMT barrel candidate
void L1MuGMTReadoutRecord::setGMTBrlCand(int nr, L1MuGMTExtendedCand const& cand) {
  if (nr>=0 && nr<4) {
    m_BarrelCands[nr] = cand;
  }
}

/// set GMT barrel candidate
void L1MuGMTReadoutRecord::setGMTBrlCand(int nr, unsigned data, unsigned rank) {
  if (nr>=0 && nr<4) {
    m_BarrelCands[nr] = L1MuGMTExtendedCand(data,rank,m_BxInEvent);
  }
}

/// set GMT forward candidate
void L1MuGMTReadoutRecord::setGMTFwdCand(int nr, L1MuGMTExtendedCand const& cand) {
  if (nr>=0 && nr<4) {
    m_ForwardCands[nr] = cand;
  }
}

/// set GMT forward candidate
void L1MuGMTReadoutRecord::setGMTFwdCand(int nr, unsigned data, unsigned rank) {
  if (nr>=0 && nr<4) {
    m_ForwardCands[nr] = L1MuGMTExtendedCand(data,rank,m_BxInEvent);
  }
}

/// set GMT candidate
void L1MuGMTReadoutRecord::setGMTCand(int nr, L1MuGMTExtendedCand const& cand) {
  if (nr>=0 && nr<4) {
    m_GMTCands[nr] = cand;
  }
}

/// set GMT candidate
void L1MuGMTReadoutRecord::setGMTCand(int nr, unsigned data) {
  if (nr>=0 && nr<4) {
    m_GMTCands[nr] = L1MuGMTExtendedCand(data,0,m_BxInEvent);
  }
}



 /// get rank of brl cand i
unsigned L1MuGMTReadoutRecord::getBrlRank(int i) const {

  return m_BarrelCands[i].rank();

}


/// get rank of fwd cand i
unsigned L1MuGMTReadoutRecord::getFwdRank(int i) const {

  return m_ForwardCands[i].rank();

}

/// set rank of brl cand i
void L1MuGMTReadoutRecord::setBrlRank(int i, unsigned value) {

  if (i>=0 && i<4) {
    m_BarrelCands[i].setRank(value);
  }  

}


/// set rank of fwd cand i
void L1MuGMTReadoutRecord::setFwdRank(int i, unsigned value) {

  if (i>=0 && i<4) {
    m_ForwardCands[i].setRank(value);
  }  

}

/// set MIP bit
void L1MuGMTReadoutRecord::setMIPbit(int eta, int phi) {

  if (phi<0 || phi > 17 || eta < 0 || eta > 13) return;

  int idx = eta * 18 + phi;
  int idx_word = idx / 32;
  int idx_bit = idx % 32;

  unsigned mask = 1 << (idx_bit-1);

  m_MIPbits[idx_word] |= mask;

}


/// set Quiet bit
void L1MuGMTReadoutRecord::setQuietbit(int eta, int phi) {

  if (phi<0 || phi > 17 || eta < 0 || eta > 13) return;

  int idx = eta * 18 + phi;
  int idx_word = idx / 32;
  int idx_bit = idx % 32;

  unsigned mask = 1 << (idx_bit-1);

  m_Quietbits[idx_word] |= mask;

}




















