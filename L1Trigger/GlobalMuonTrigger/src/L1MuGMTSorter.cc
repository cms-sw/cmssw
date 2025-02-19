//-------------------------------------------------
//
//   Class: L1MuGMTSorter
//
//   Description: GMT Muon Sorter
//
//
//   $Date: 2007/04/10 09:59:19 $
//   $Revision: 1.5 $
//
//   Author :
//   N. Neumeister             CERN EP
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTSorter.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPSB.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMerger.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTSorter
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuGMTSorter::L1MuGMTSorter(const L1MuGlobalMuonTrigger& gmt) :
      m_gmt(gmt), m_MuonCands() {

  m_MuonCands.reserve(4);
  
}


//--------------
// Destructor --
//--------------

L1MuGMTSorter::~L1MuGMTSorter() {

}


//--------------
// Operations --
//--------------

//
// run GMT Sorter 
//
void L1MuGMTSorter::run() {

  std::vector<L1MuGMTExtendedCand*> mycands;
   std::vector<L1MuGMTExtendedCand*> my_brl_cands;

  std::vector<L1MuGMTExtendedCand*>::const_iterator iter;
  
  // get muon candidates from barrel Merger 
  const std::vector<L1MuGMTExtendedCand*>& brl_cands = m_gmt.Merger(0)->Cands();
  iter = brl_cands.begin();
  while ( iter != brl_cands.end() ) {
    if ( *iter && !(*iter)->empty() ) {
      my_brl_cands.push_back((*iter));
    }
    iter++;
  }
  // sort by rank
  stable_sort( my_brl_cands.begin(), my_brl_cands.end(), L1MuGMTExtendedCand::Rank() );

  // copy best four of brl to main sorter
  iter = my_brl_cands.begin();
  int count=0;
  while ( iter != my_brl_cands.end() && (count<4) ) {
    if ( *iter && !(*iter)->empty() ) {
      mycands.push_back((*iter));
      m_gmt.DebugBlockForFill()->SetBrlGMTCands( count, **iter) ; 
      m_gmt.currentReadoutRecord()->setGMTBrlCand ( count, **iter );
      count++;
   }
    iter++;
  }

   
  std::vector<L1MuGMTExtendedCand*> my_fwd_cands;

  // get muon candidates from forward Merger 
  const std::vector<L1MuGMTExtendedCand*>& fwd_cands = m_gmt.Merger(1)->Cands();
  iter = fwd_cands.begin();
  while ( iter != fwd_cands.end() ) {
    if ( *iter && !(*iter)->empty() ) {
      my_fwd_cands.push_back((*iter));
    }
    iter++;
  }
   // sort by rank
  stable_sort( my_fwd_cands.begin(), my_fwd_cands.end(), L1MuGMTExtendedCand::Rank() );


  // copy best four of fwd to main sorter
  iter = my_fwd_cands.begin();
  count=0;
  while ( iter != my_fwd_cands.end() && (count<4) ) {
    if ( *iter && !(*iter)->empty() ) {
      mycands.push_back((*iter));
      m_gmt.DebugBlockForFill()->SetFwdGMTCands( count, **iter) ; 
      m_gmt.currentReadoutRecord()->setGMTFwdCand ( count, **iter );
      count++;
    }
    iter++;
  }


  // print input data
  if ( L1MuGMTConfig::Debug(5) ) {
    edm::LogVerbatim("GMT_Sorter_info") << "GMT Sorter input: "
         << mycands.size();
    std::vector<L1MuGMTExtendedCand*>::const_iterator iter;
    for ( iter = mycands.begin(); iter != mycands.end(); iter++ ) {
      if (*iter ) (*iter)->print();
    }
  }
  
  // sort by rank
  stable_sort( mycands.begin(), mycands.end(), L1MuGMTExtendedCand::Rank() );

  // copy the best 4 candidates
  int number_of_cands = 0;
  std::vector<L1MuGMTExtendedCand*>::const_iterator iter1 = mycands.begin();
  while ( iter1 != mycands.end() ) {
    if ( *iter1 && number_of_cands < 4 ) {
      m_MuonCands.push_back(*iter1);
      m_gmt.currentReadoutRecord()->setGMTCand ( count, **iter1 );
      number_of_cands++;
    }
    iter1++;
  }  

} 


//
// reset GMT Sorter
//
void L1MuGMTSorter::reset() {

  std::vector<const L1MuGMTExtendedCand*>::iterator iter;
  for ( iter = m_MuonCands.begin(); iter != m_MuonCands.end(); iter++ ) {
    *iter = 0;
  }
  m_MuonCands.clear();

} 


//
// print GMT sorter results
//
void L1MuGMTSorter::print() {

  edm::LogVerbatim("GMT_Sorter_info") << " ";
  edm::LogVerbatim("GMT_Sorter_info") << "Muon candidates found by the L1 Global Muon Trigger : "
       << numberOfCands();
  std::vector<const L1MuGMTExtendedCand*>::const_iterator iter = m_MuonCands.begin();
  while ( iter != m_MuonCands.end() ) {
    if ( *iter ) (*iter)->print();
    iter++;
  }
  edm::LogVerbatim("GMT_Sorter_info") << " ";

}


















