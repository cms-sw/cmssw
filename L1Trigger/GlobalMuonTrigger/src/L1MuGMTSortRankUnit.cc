//-------------------------------------------------
//
//   Class: L1MuGMTSortRankUnit
//
//   Description: GMT Sort Rank Unit
//
//
//   $Date $
//   $Revision $
//
//   Author :
//   H. Sakulin                   HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTSortRankUnit.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankEtaQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankPtQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankEtaPhiLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFSortRankCombineLUT.h"

  
//---------------------------------
//       class L1MuGMTSortRankUnit
//---------------------------------

unsigned L1MuGMTSortRankUnit::sort_rank(const L1MuRegionalCand* muon) {

  if ( muon == 0 || muon->empty() ) return 0;

  unsigned lut_idx= muon->type_idx();
  
  // obtain inputs as coded in HW
  unsigned eta = muon->eta_packed();
  unsigned q = muon->quality_packed();
  unsigned pt  = muon->pt_packed();
  unsigned phi = muon->phi_packed();

  // lookup eta-q
  L1MuGMTLFSortRankEtaQLUT* etaq_lut = L1MuGMTConfig::getLFSortRankEtaQLUT();
  unsigned rank_etaq = etaq_lut->SpecificLookup_rank_etaq (lut_idx, eta, q);
  
  // lookup pt-q
  L1MuGMTLFSortRankPtQLUT* ptq_lut = L1MuGMTConfig::getLFSortRankPtQLUT();
  unsigned rank_ptq = ptq_lut->SpecificLookup_rank_ptq (lut_idx, q, pt);
  
  // lookup etaphi
  L1MuGMTLFSortRankEtaPhiLUT* etaphi_lut = L1MuGMTConfig::getLFSortRankEtaPhiLUT();
  unsigned rank_etaphi = etaphi_lut->SpecificLookup_rank_etaphi (lut_idx, eta, phi);
  
  // combine
  L1MuGMTLFSortRankCombineLUT* combine_lut = L1MuGMTConfig::getLFSortRankCombineLUT();
  unsigned rank = combine_lut->SpecificLookup_sort_rank (lut_idx, rank_etaq, rank_ptq, rank_etaphi);

  return rank;
}

unsigned L1MuGMTSortRankUnit::getVeryLowQualityLevel(const L1MuRegionalCand* muon) {

  if ( muon == 0 || muon->empty() ) return 0;

  unsigned lut_idx= muon->type_idx();
  
  // obtain inputs as coded in HW
  unsigned eta = muon->eta_packed();
  unsigned q = muon->quality_packed();

  // lookup eta-q
  L1MuGMTLFSortRankEtaQLUT* etaq_lut = L1MuGMTConfig::getLFSortRankEtaQLUT();
  return  etaq_lut->SpecificLookup_vlq (lut_idx, eta, q);
}

bool L1MuGMTSortRankUnit::isDisabled(const L1MuRegionalCand* muon) {

  if ( muon == 0 || muon->empty() ) return 1;

  unsigned lut_idx= muon->type_idx();
  
  // obtain inputs as coded in HW
  unsigned eta = muon->eta_packed();
  unsigned phi = muon->phi_packed();

  // lookup eta-q
  L1MuGMTLFSortRankEtaPhiLUT* etaphi_lut = L1MuGMTConfig::getLFSortRankEtaPhiLUT();

  return  etaphi_lut->SpecificLookup_rank_etaphi (lut_idx, eta, phi) == 3;
}
