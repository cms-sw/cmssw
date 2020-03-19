//-------------------------------------------------
//
//   Class: L1MuGMTMerger
//
//   Description:  GMT Merger
//
//
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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMerger.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <cmath>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPSB.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTSortRankUnit.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatcher.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTCancelOutUnit.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMipIsoAU.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankPtQLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankEtaPhiLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMergeRankCombineLUT.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFEtaConvLUT.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPhiProLUT.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFPtMixLUT.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTReg.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//---------------------------------
//       class L1MuGMTMerger
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTMerger::L1MuGMTMerger(const L1MuGlobalMuonTrigger& gmt, int id) : m_gmt(gmt), m_id(id), dtcsc_mu(4), rpc_mu(4) {
  dtcsc_mu.reserve(4);
  rpc_mu.reserve(4);
}

//--------------
// Destructor --
//--------------
L1MuGMTMerger::~L1MuGMTMerger() { reset(); }

//--------------
// Operations --
//--------------

//
// run Merger
//
void L1MuGMTMerger::run() {
  load();
  merge();
}

//
// reset Merger
//
void L1MuGMTMerger::reset() {
  for (int i = 0; i < 4; i++) {
    dtcsc_mu[i] = nullptr;
    rpc_mu[i] = nullptr;
  }

  std::vector<L1MuGMTExtendedCand*>::iterator iter;
  for (iter = m_MuonCands.begin(); iter != m_MuonCands.end(); iter++) {
    if (*iter)
      delete (*iter);
    *iter = nullptr;
  }

  m_MuonCands.clear();
}

//
// print selection results
//
void L1MuGMTMerger::print() const {
  edm::LogVerbatim("GMT_Merger_info") << " ";

  std::vector<L1MuGMTExtendedCand*>::const_iterator iter;
  for (iter = m_MuonCands.begin(); iter != m_MuonCands.end(); iter++) {
    if (*iter && !(*iter)->empty())
      (*iter)->print();
  }

  edm::LogVerbatim("GMT_Merger_info") << " ";
}

//
// load Merger (get data from PSB)
//
void L1MuGMTMerger::load() {
  // barrel Merger gets DTBX and barrel RPC muons
  if (m_id == 0) {
    for (unsigned idt = 0; idt < L1MuGMTConfig::MAXDTBX; idt++) {
      dtcsc_mu[idt] = m_gmt.Data()->DTBXMuon(idt);
    }
    for (unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCbarrel; irpc++) {
      rpc_mu[irpc] = m_gmt.Data()->RPCMuon(irpc);
    }
  }

  // endcap Merger gets CSC and endcap RPC muons
  if (m_id == 1) {
    for (unsigned icsc = 0; icsc < L1MuGMTConfig::MAXCSC; icsc++) {
      dtcsc_mu[icsc] = m_gmt.Data()->CSCMuon(icsc);
    }
    for (unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCendcap; irpc++) {
      rpc_mu[irpc] = m_gmt.Data()->RPCMuon(irpc + 4);
    }
  }
}

void L1MuGMTMerger::merge() {
  const L1MuGMTMatrix<bool>& pairM = m_gmt.Matcher(m_id)->pairM();

  // Handling of cancel-out and empty muons is different in software and hardware
  //
  // - in hardware, a candidate may be empty (pt_code == 0) and an empty bit
  //   is passed to the sorter in order to suppress the candidate. In the
  //   software no candidate is created in this case.
  //
  // - in hardware, RPC candidates are passed to the sorter even if they are
  //   also used in a matched pair. They are then suppressed in the sorter. In
  //   software no RPC candidate is created if the muon is used in a pair.
  //
  // - in hardware, cancel-out signals from the cancel-out units in the own and
  //   other Logic FPGA are passed to the sorter in order to cancel out muons.
  //   In software the cancel-out signals are alrady checked here in the merger
  //   and no candidates are created for cancelled muons.
  //
  // There may therefore be less muons passed to the sorter in software than
  // in Hardware. At the output of the first sorter stage the results should
  // be comparable, again.

  unsigned HaloOverwritesMatched = 1;

  // loop over DT/CSC muons
  for (int i = 0; i < 4; i++) {
    if (dtcsc_mu[i] != nullptr) {
      int match_idx = pairM.rowAny(i);

      int csc_is_halo = (m_id == 1) && (dtcsc_mu[i]->finehalo_packed() == 1);

      if ((match_idx != -1) &&  // is it matched?
          (!(csc_is_halo && HaloOverwritesMatched)))
        createMergedCand(i, match_idx);
      else {
        // check my first and the other chip's second cancel-out units
        if ((!m_gmt.CancelOutUnit(m_id)->cancelMyChipMuon(i)) &&
            (!m_gmt.CancelOutUnit(3 - m_id)->cancelOtherChipMuon(i)) && (!L1MuGMTSortRankUnit::isDisabled(dtcsc_mu[i])))
          createDTCSCCand(i);
      }
    }
  }

  // additionally loop over RPC muons
  for (int j = 0; j < 4; j++) {
    if (rpc_mu[j] != nullptr) {
      int match_idx = pairM.colAny(j);

      if (match_idx == -1) {  // is it unmatched?
        if ((!m_gmt.CancelOutUnit(m_id + 2)->cancelMyChipMuon(j)) && (!L1MuGMTSortRankUnit::isDisabled(rpc_mu[j])))
          createRPCCand(j);
      }
    }
  }

  // set physical values in the GMT candidates for use in the analysis
  const L1MuTriggerScales* theTriggerScales = L1MuGMTConfig::getTriggerScales();
  const L1MuTriggerPtScale* theTriggerPtScale = L1MuGMTConfig::getTriggerPtScale();

  std::vector<L1MuGMTExtendedCand*>::const_iterator icand;
  for (icand = m_MuonCands.begin(); icand != m_MuonCands.end(); icand++) {
    L1MuGMTExtendedCand* cand = (*icand);
    cand->setPhiValue(theTriggerScales->getPhiScale()->getLowEdge(cand->phiIndex()));
    cand->setEtaValue(theTriggerScales->getGMTEtaScale()->getCenter(cand->etaIndex()));
    cand->setPtValue(theTriggerPtScale->getPtScale()->getLowEdge(cand->ptIndex()));
    // cand->setPtValue( theTriggerScales->getPtScale()->getLowEdge( cand->ptIndex() ));
  }
}

void L1MuGMTMerger::createDTCSCCand(int idx_dtcsc) {
  L1MuGMTExtendedCand* tmpmuon = new L1MuGMTExtendedCand();

  tmpmuon->setBx(dtcsc_mu[idx_dtcsc]->bx());
  tmpmuon->setPhiPacked(projectedPhi(dtcsc_mu[idx_dtcsc]));
  tmpmuon->setEtaPacked(convertedEta(dtcsc_mu[idx_dtcsc]));
  tmpmuon->setPtPacked(dtcsc_mu[idx_dtcsc]->pt_packed());
  tmpmuon->setChargePacked(sysign(dtcsc_mu[idx_dtcsc]));
  tmpmuon->setMIP(m_gmt.MipIsoAU(m_id)->MIP(idx_dtcsc));
  tmpmuon->setIsolation(m_gmt.MipIsoAU(m_id)->ISO(idx_dtcsc));
  tmpmuon->setRank(L1MuGMTSortRankUnit::sort_rank(dtcsc_mu[idx_dtcsc]));

  unsigned quality = 0;
  switch (L1MuGMTSortRankUnit::getVeryLowQualityLevel(dtcsc_mu[idx_dtcsc])) {
    case 0:
      quality = 6;
      break;  //DT/CSC
    case 1:
      quality = 2;
      break;  //VERY LOW QUALITY LEVEL 1
    case 2:
      quality = 3;
      break;  //VERY LOW QUALITY LEVEL 2
    case 3:
      quality = 4;
      break;  //VERY LOW QUALITY LEVEL 3
  }

  if ((m_id == 1) && (dtcsc_mu[idx_dtcsc]->finehalo_packed() == 1))
    quality = 1;  // HALO quality

  tmpmuon->setQuality(quality);  // RPC
  tmpmuon->setDTCSCIndex(idx_dtcsc);
  tmpmuon->setRPCIndex(0);
  tmpmuon->setFwdBit(m_id);
  tmpmuon->setRPCBit(0);

  m_MuonCands.push_back(tmpmuon);
}

void L1MuGMTMerger::createRPCCand(int idx_rpc) {
  L1MuGMTExtendedCand* tmpmuon = new L1MuGMTExtendedCand();

  tmpmuon->setBx(rpc_mu[idx_rpc]->bx());
  tmpmuon->setPhiPacked(projectedPhi(rpc_mu[idx_rpc]));
  tmpmuon->setEtaPacked(convertedEta(rpc_mu[idx_rpc]));
  tmpmuon->setPtPacked(rpc_mu[idx_rpc]->pt_packed());
  tmpmuon->setChargePacked(sysign(rpc_mu[idx_rpc]));
  tmpmuon->setMIP(m_gmt.MipIsoAU(m_id)->MIP(idx_rpc + 4));
  tmpmuon->setIsolation(m_gmt.MipIsoAU(m_id)->ISO(idx_rpc + 4));
  tmpmuon->setRank(L1MuGMTSortRankUnit::sort_rank(rpc_mu[idx_rpc]));

  unsigned quality = 0;
  switch (L1MuGMTSortRankUnit::getVeryLowQualityLevel(rpc_mu[idx_rpc])) {
    case 0:
      quality = 5;
      break;  //RPC
    case 1:
      quality = 2;
      break;  //VERY LOW QUALITY LEVEL1
    case 2:
      quality = 3;
      break;  //VERY LOW QUALITY LEVEL2
    case 3:
      quality = 4;
      break;  //VERY LOW QUALITY LEVEL3
  }

  tmpmuon->setQuality(quality);  // RPC
  tmpmuon->setDTCSCIndex(0);
  tmpmuon->setRPCIndex(idx_rpc);
  tmpmuon->setFwdBit(m_id);
  tmpmuon->setRPCBit(1);

  m_MuonCands.push_back(tmpmuon);
}

int L1MuGMTMerger::selectDTCSC(unsigned MMconfig, int by_rank, int by_pt, int by_combi) const {
  return ((MMconfig & 32) == 32) || (((MMconfig & 8) == 8) && by_rank) || (((MMconfig & 4) == 4) && by_pt) ||
         (((MMconfig & 2) == 2) && by_combi);
}

int L1MuGMTMerger::doSpecialMerge(unsigned MMconfig) const { return (MMconfig & 1) == 1; }

int L1MuGMTMerger::doANDMerge(unsigned MMconfig) const { return (MMconfig & 64) == 64; }

unsigned L1MuGMTMerger::convertedEta(const L1MuRegionalCand* mu) const {
  L1MuGMTLFEtaConvLUT* etaconv_lut = L1MuGMTConfig::getLFEtaConvLUT();
  return etaconv_lut->SpecificLookup_eta_gmt(mu->type_idx(), mu->eta_packed());
}

unsigned L1MuGMTMerger::projectedPhi(const L1MuRegionalCand* mu) const {
  // convert eta
  L1MuGMTLFPhiProEtaConvLUT* phiproetaconv_lut = L1MuGMTConfig::getLFPhiProEtaConvLUT();
  unsigned eta4 = phiproetaconv_lut->SpecificLookup_eta_out(mu->type_idx(), mu->eta_packed());

  // look up delta-phi 9 bit signed
  L1MuGMTLFPhiProLUT* phipro_lut = L1MuGMTConfig::getLFPhiProLUT();
  unsigned dphi9 = phipro_lut->SpecificLookup_dphi(mu->type_idx(), eta4, mu->pt_packed(), mu->charge_packed());

  // sign extend
  L1MuSignedPacking<9> DPhiPacking;
  int dphi = DPhiPacking.idxFromPacked(dphi9);

  // add modulo 144
  int newphi = mu->phi_packed() + dphi;
  if (newphi < 0)
    newphi += 144;
  if (newphi >= 144)
    newphi -= 144;

  return (unsigned)newphi;
}

unsigned L1MuGMTMerger::sysign(const L1MuRegionalCand* mu) const {
  unsigned sysign = mu->charge_packed();

  if (mu->charge_valid_packed() == 0)
    sysign = 2;  // undefined charge

  return sysign;
}

void L1MuGMTMerger::createMergedCand(int idx_dtcsc, int idx_rpc) {
  // In the hardware matrices of select_bits are calculated for all
  // possible pairings.
  // In ORCA we only calculate selec-bits for the actual
  // pairs to save time.

  // look up merge ranks
  int merge_rank_dtcsc = merge_rank(dtcsc_mu[idx_dtcsc]);
  int merge_rank_rpc = merge_rank(rpc_mu[idx_rpc]);

  // calculate select-bits (1: take DT/CSC, 0: take RPC)
  // fix: if equal prefer DT/CSC as in HW!
  //  int selected_by_rank = abs(merge_rank_dtcsc) > abs(merge_rank_rpc);
  int selected_by_rank = abs(merge_rank_dtcsc) >= abs(merge_rank_rpc);
  int selected_by_pt = dtcsc_mu[idx_dtcsc]->pt_packed() <= rpc_mu[idx_rpc]->pt_packed();

  // Selection by combination of min pt and higher rank
  // select by rank if both flags are set, otherwise by min pt
  // in other words: select by minpt if one flag is not set
  int selected_by_combi = (merge_rank_dtcsc < 0 && merge_rank_rpc < 0) ? selected_by_rank : selected_by_pt;

  L1MuGMTExtendedCand* tmpmuon = new L1MuGMTExtendedCand();

  tmpmuon->setBx(dtcsc_mu[idx_dtcsc]->bx());

  // merge phi
  //  unsigned MMConfig_phi = 32; // take DT
  unsigned MMConfig_phi = L1MuGMTConfig::getRegMMConfigPhi()->getValue(m_id);

  unsigned phi = 0;

  if (selectDTCSC(MMConfig_phi, selected_by_rank, selected_by_pt, selected_by_combi))
    phi = projectedPhi(dtcsc_mu[idx_dtcsc]);
  else
    phi = projectedPhi(rpc_mu[idx_rpc]);

  tmpmuon->setPhiPacked(phi);

  // merge eta
  unsigned MMConfig_eta = L1MuGMTConfig::getRegMMConfigEta()->getValue(m_id);

  unsigned eta = 0;

  if (doSpecialMerge(MMConfig_eta)) {
    if ((m_id == 1) || dtcsc_mu[idx_dtcsc]->finehalo_packed())
      eta = convertedEta(dtcsc_mu[idx_dtcsc]);
    else
      eta = convertedEta(rpc_mu[idx_rpc]);
  } else {
    if (selectDTCSC(MMConfig_eta, selected_by_rank, selected_by_pt, selected_by_combi))
      eta = convertedEta(dtcsc_mu[idx_dtcsc]);
    else
      eta = convertedEta(rpc_mu[idx_rpc]);
  }
  tmpmuon->setEtaPacked(eta);

  // merge pt
  unsigned MMConfig_pt = L1MuGMTConfig::getRegMMConfigPt()->getValue(m_id);

  unsigned pt = 0;

  if (doSpecialMerge(MMConfig_pt)) {  // mix pt
    L1MuGMTLFPtMixLUT* ptmix_lut = L1MuGMTConfig::getLFPtMixLUT();
    pt = ptmix_lut->SpecificLookup_pt_mixed(m_id, dtcsc_mu[idx_dtcsc]->pt_packed(), rpc_mu[idx_rpc]->pt_packed());
  } else {
    if (selectDTCSC(MMConfig_pt, selected_by_rank, selected_by_pt, selected_by_combi))
      pt = dtcsc_mu[idx_dtcsc]->pt_packed();
    else
      pt = rpc_mu[idx_rpc]->pt_packed();
  }
  tmpmuon->setPtPacked(pt);

  // merge charge
  unsigned MMConfig_charge = L1MuGMTConfig::getRegMMConfigCharge()->getValue(m_id);

  unsigned sy_sign = 0;

  if (doSpecialMerge(MMConfig_charge)) {
    // based on charge valid bits
    if (rpc_mu[idx_rpc]->charge_valid_packed() == 1 && dtcsc_mu[idx_dtcsc]->charge_valid_packed() == 0)
      sy_sign = sysign(rpc_mu[idx_rpc]);
    else
      sy_sign = sysign(dtcsc_mu[idx_dtcsc]);
  } else {
    if (selectDTCSC(MMConfig_charge, selected_by_rank, selected_by_pt, selected_by_combi))
      sy_sign = sysign(dtcsc_mu[idx_dtcsc]);
    else
      sy_sign = sysign(rpc_mu[idx_rpc]);
  }
  tmpmuon->setChargePacked(sy_sign);

  // merge quality

  // merge MIP
  unsigned MMConfig_MIP = L1MuGMTConfig::getRegMMConfigMIP()->getValue(m_id);

  bool mip_bit = false;

  bool mip_bit_dtcsc = m_gmt.MipIsoAU(m_id)->MIP(idx_dtcsc);
  bool mip_bit_rpc = m_gmt.MipIsoAU(m_id)->MIP(idx_rpc + 4);

  if (doSpecialMerge(MMConfig_MIP)) {
    if (doANDMerge(MMConfig_MIP))
      mip_bit = mip_bit_dtcsc && mip_bit_rpc;
    else
      mip_bit = mip_bit_dtcsc || mip_bit_rpc;
  } else {
    if (selectDTCSC(MMConfig_MIP, selected_by_rank, selected_by_pt, selected_by_combi))
      mip_bit = mip_bit_dtcsc;
    else
      mip_bit = mip_bit_rpc;
  }

  tmpmuon->setMIP(mip_bit);

  // merge ISO
  unsigned MMConfig_ISO = L1MuGMTConfig::getRegMMConfigISO()->getValue(m_id);

  bool iso_bit = false;

  bool iso_bit_dtcsc = m_gmt.MipIsoAU(m_id)->ISO(idx_dtcsc);
  bool iso_bit_rpc = m_gmt.MipIsoAU(m_id)->ISO(idx_rpc + 4);

  if (doSpecialMerge(MMConfig_ISO)) {
    if (doANDMerge(MMConfig_ISO))
      iso_bit = iso_bit_dtcsc && iso_bit_rpc;
    else
      iso_bit = iso_bit_dtcsc || iso_bit_rpc;
  } else {
    if (selectDTCSC(MMConfig_ISO, selected_by_rank, selected_by_pt, selected_by_combi))
      iso_bit = iso_bit_dtcsc;
    else
      iso_bit = iso_bit_rpc;
  }

  tmpmuon->setIsolation(iso_bit);

  // merge sort rank
  unsigned MMConfig_SRK = L1MuGMTConfig::getRegMMConfigSRK()->getValue(m_id);

  unsigned rank_offset = L1MuGMTConfig::getRegSortRankOffset()->getValue(m_id);

  unsigned rank = 0;
  if (selectDTCSC(MMConfig_SRK, selected_by_rank, selected_by_pt, selected_by_combi))
    rank = L1MuGMTSortRankUnit::sort_rank(dtcsc_mu[idx_dtcsc]) + rank_offset;
  else
    rank = L1MuGMTSortRankUnit::sort_rank(rpc_mu[idx_rpc]) + rank_offset;

  tmpmuon->setRank(rank);

  // quality of merged candidate
  tmpmuon->setQuality(7);  // code for matched muons

  tmpmuon->setDTCSCIndex(idx_dtcsc);
  tmpmuon->setRPCIndex(idx_rpc);
  tmpmuon->setFwdBit(m_id);
  tmpmuon->setRPCBit(0);

  m_MuonCands.push_back(tmpmuon);
}

// calculate merge rank as in HW

int L1MuGMTMerger::merge_rank(const L1MuRegionalCand* muon) const {
  if (muon == nullptr || muon->empty())
    return 0;

  unsigned lut_idx = muon->type_idx();

  // obtain inputs as coded in HW
  unsigned eta = muon->eta_packed();
  unsigned q = muon->quality_packed();
  unsigned pt = muon->pt_packed();
  unsigned phi = muon->phi_packed();

  // lookup eta-q
  L1MuGMTLFMergeRankEtaQLUT* etaq_lut = L1MuGMTConfig::getLFMergeRankEtaQLUT();
  unsigned rank_etaq = etaq_lut->SpecificLookup_rank_etaq(lut_idx, eta, q);
  unsigned flag = etaq_lut->SpecificLookup_flag(lut_idx, eta, q);

  // lookup pt-q
  L1MuGMTLFMergeRankPtQLUT* ptq_lut = L1MuGMTConfig::getLFMergeRankPtQLUT();
  unsigned rank_ptq = ptq_lut->SpecificLookup_rank_ptq(lut_idx, q, pt);

  // lookup etaphi
  L1MuGMTLFMergeRankEtaPhiLUT* etaphi_lut = L1MuGMTConfig::getLFMergeRankEtaPhiLUT();
  unsigned rank_etaphi = etaphi_lut->SpecificLookup_rank_etaphi(lut_idx, eta, phi);

  // combine
  L1MuGMTLFMergeRankCombineLUT* combine_lut = L1MuGMTConfig::getLFMergeRankCombineLUT();
  unsigned rank = combine_lut->SpecificLookup_merge_rank(lut_idx, rank_etaq, rank_ptq, rank_etaphi);

  int rank_signed = rank;

  if (flag == 1)
    rank_signed *= -1;

  return rank_signed;
}
