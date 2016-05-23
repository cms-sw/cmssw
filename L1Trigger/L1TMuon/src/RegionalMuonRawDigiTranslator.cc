#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"

void
l1t::RegionalMuonRawDigiTranslator::fillRegionalMuonCand(RegionalMuonCand& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63, int proc, tftype tf)
{
  // translations as defined in DN-15-017
  mu.setHwPt((raw_data_00_31 >> ptShift_) & ptMask_);
  mu.setHwQual((raw_data_00_31 >> qualShift_) & qualMask_); 

  // eta is coded as two's complement
  int abs_eta = (raw_data_00_31 >> absEtaShift_) & absEtaMask_;
  if ((raw_data_00_31 >> etaSignShift_) & 0x1) {
     mu.setHwEta(abs_eta - (1 << (etaSignShift_ - absEtaShift_)));
  } else {
     mu.setHwEta(abs_eta);
  }

  // phi is coded as two's complement
  int abs_phi = (raw_data_00_31 >> absPhiShift_) & absPhiMask_;
  if ((raw_data_00_31 >> phiSignShift_) & 0x1) {
     mu.setHwPhi(abs_phi - (1 << (phiSignShift_ - absPhiShift_)));
  } else {
     mu.setHwPhi(abs_phi);
  }

  // sign is coded as -1^signBit
  mu.setHwSign((raw_data_32_63 >> signShift_) & 0x1);
  mu.setHwSignValid((raw_data_32_63 >> signValidShift_) & 0x1);
  mu.setHwHF((raw_data_00_31 >> hfShift_) & hfMask_);

  // set track address with subaddresses
  int rawTrackAddress = (raw_data_32_63 >> trackAddressShift_) & trackAddressMask_;
  if (tf == bmtf) {
    int segSel = (rawTrackAddress >> bmtfTrAddrSegSelShift_) & bmtfTrAddrSegSelMask_;
    int detSide = (rawTrackAddress >> bmtfTrAddrDetSideShift_) & 0x1;
    int wheelNum = (rawTrackAddress >> bmtfTrAddrWheelShift_) & bmtfTrAddrWheelMask_;
    int statAddr1 = ((rawTrackAddress >> bmtfTrAddrStat1Shift_) & bmtfTrAddrStat1Mask_)
                  | ((segSel & 0x1) << 2);
    int statAddr2 = ((rawTrackAddress >> bmtfTrAddrStat2Shift_) & bmtfTrAddrStat2Mask_)
                  | ((segSel & 0x2) << 3);
    int statAddr3 = ((rawTrackAddress >> bmtfTrAddrStat3Shift_) & bmtfTrAddrStat3Mask_)
                  | ((segSel & 0x4) << 2);
    int statAddr4 = ((rawTrackAddress >> bmtfTrAddrStat4Shift_) & bmtfTrAddrStat4Mask_)
                  | ((segSel & 0x8) << 1);
    mu.setTrackSubAddress(RegionalMuonCand::kWheelSide, detSide);
    mu.setTrackSubAddress(RegionalMuonCand::kWheelNum, wheelNum);
    mu.setTrackSubAddress(RegionalMuonCand::kStat1, statAddr1);
    mu.setTrackSubAddress(RegionalMuonCand::kStat2, statAddr2);
    mu.setTrackSubAddress(RegionalMuonCand::kStat3, statAddr3);
    mu.setTrackSubAddress(RegionalMuonCand::kStat4, statAddr4);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat1, segSel & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat2, segSel & 0x2);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat3, segSel & 0x4);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat4, segSel & 0x8);
  } else if (tf == emtf_neg || tf == emtf_pos) {
    mu.setTrackSubAddress(RegionalMuonCand::kSectorId, (rawTrackAddress >> emtfTrAddrSectIdShift_) & emtfTrAddrSectIdMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME1, (rawTrackAddress >> emtfTrAddrMe1Shift_) & emtfTrAddrMe1Mask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME1SubSecId, (rawTrackAddress >> emtfTrAddrMe1SubSecIdShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME1Order, (rawTrackAddress >> emtfTrAddrMe1OrderShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME2, (rawTrackAddress >> emtfTrAddrMe2Shift_) & emtfTrAddrMe2Mask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME2SubSecId, (rawTrackAddress >> emtfTrAddrMe2SubSecIdShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME2Order, (rawTrackAddress >> emtfTrAddrMe2OrderShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME3, (rawTrackAddress >> emtfTrAddrMe3Shift_) & emtfTrAddrMe3Mask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME3SubSecId, (rawTrackAddress >> emtfTrAddrMe3SubSecIdShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME3Order, (rawTrackAddress >> emtfTrAddrMe3OrderShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME4, (rawTrackAddress >> emtfTrAddrMe4Shift_) & emtfTrAddrMe4Mask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME4SubSecId, (rawTrackAddress >> emtfTrAddrMe4SubSecIdShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME4Order, (rawTrackAddress >> emtfTrAddrMe4OrderShift_) & 0x1);
  } else {
    std::map<int, int> trackAddr;
    trackAddr[0] = rawTrackAddress;
    mu.setTrackAddress(trackAddr);
  }

  mu.setTFIdentifiers(proc, tf);
  mu.setDataword(raw_data_32_63, raw_data_00_31);
}

void
l1t::RegionalMuonRawDigiTranslator::fillRegionalMuonCand(RegionalMuonCand& mu, uint64_t dataword, int proc, tftype tf)
{
  fillRegionalMuonCand(mu, (uint32_t)(dataword & 0xFFFFFFFF), (uint32_t)((dataword >> 32) & 0xFFFFFFFF), proc, tf);
}

void
l1t::RegionalMuonRawDigiTranslator::generatePackedDataWords(const RegionalMuonCand& mu, uint32_t &raw_data_00_31, uint32_t &raw_data_32_63)
{
  int abs_eta = mu.hwEta();
  if (abs_eta < 0) {
    abs_eta += (1 << (etaSignShift_ - absEtaShift_));
  }
  int abs_phi = mu.hwPhi();
  if (abs_phi < 0) {
    abs_phi += (1 << (phiSignShift_ - absPhiShift_));
  }
  raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_
                 | (mu.hwQual() & qualMask_) << qualShift_
                 | (abs_eta & absEtaMask_) << absEtaShift_
                 | (mu.hwEta() < 0) << etaSignShift_
                 | (mu.hwHF() & hfMask_) << hfShift_
                 | (abs_phi & absPhiMask_) << absPhiShift_
                 | (mu.hwPhi() < 0) << phiSignShift_;

  // generate the raw track address from the subaddresses
  int tf = mu.trackFinderType();
  int rawTrkAddr = 0;
  if (tf == bmtf) {
    // protection against a track address map with the wrong size
    if (mu.trackAddress().size() == RegionalMuonCand::kNumBmtfSubAddr) {
      int detSide = mu.trackSubAddress(RegionalMuonCand::kWheelSide);
      int wheelNum = mu.trackSubAddress(RegionalMuonCand::kWheelNum);
      int stat1 = mu.trackSubAddress(RegionalMuonCand::kStat1);
      int stat2 = mu.trackSubAddress(RegionalMuonCand::kStat2);
      int stat3 = mu.trackSubAddress(RegionalMuonCand::kStat3);
      int stat4 = mu.trackSubAddress(RegionalMuonCand::kStat4);

      int segSel = mu.trackSubAddress(RegionalMuonCand::kSegSelStat1)
                 | (mu.trackSubAddress(RegionalMuonCand::kSegSelStat2)) >> 1
                 | (mu.trackSubAddress(RegionalMuonCand::kSegSelStat3)) >> 2
                 | (mu.trackSubAddress(RegionalMuonCand::kSegSelStat4)) >> 3;

      rawTrkAddr = (segSel & bmtfTrAddrSegSelMask_) << bmtfTrAddrSegSelShift_
                 | (detSide & 0x1) << bmtfTrAddrDetSideShift_
                 | (wheelNum & bmtfTrAddrWheelMask_) << bmtfTrAddrWheelShift_
                 | (stat1 & bmtfTrAddrStat1Mask_) << bmtfTrAddrStat1Shift_
                 | (stat2 & bmtfTrAddrStat2Mask_) << bmtfTrAddrStat2Shift_
                 | (stat3 & bmtfTrAddrStat3Mask_) << bmtfTrAddrStat3Shift_
                 | (stat4 & bmtfTrAddrStat4Mask_) << bmtfTrAddrStat4Shift_;
    } else {
      edm::LogWarning("L1T") << "BMTF muon track address map contains " << mu.trackAddress().size() << " instead of the expected " << RegionalMuonCand::kNumBmtfSubAddr << " subaddresses. Check the data format. Setting track address to 0.";
      rawTrkAddr = 0;
    }
  } else if (tf == emtf_neg || tf == emtf_pos) {
    // protection against a track address map with the wrong size
    if (mu.trackAddress().size() == RegionalMuonCand::kNumEmtfSubAddr) {
      rawTrkAddr = (mu.trackSubAddress(RegionalMuonCand::kSectorId) & emtfTrAddrSectIdMask_) << emtfTrAddrSectIdShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME1) & emtfTrAddrMe1Mask_) << emtfTrAddrMe1Shift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME1SubSecId) & 0x1) << emtfTrAddrMe1SubSecIdShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME1Order) & 0x1) << emtfTrAddrMe1OrderShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME2) & emtfTrAddrMe2Mask_) << emtfTrAddrMe2Shift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME2SubSecId) & 0x1) << emtfTrAddrMe2SubSecIdShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME2Order) & 0x1) << emtfTrAddrMe2OrderShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME3) & emtfTrAddrMe3Mask_) << emtfTrAddrMe3Shift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME3SubSecId) & 0x1) << emtfTrAddrMe3SubSecIdShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME3Order) & 0x1) << emtfTrAddrMe3OrderShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME4) & emtfTrAddrMe4Mask_) << emtfTrAddrMe4Shift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME4SubSecId) & 0x1) << emtfTrAddrMe4SubSecIdShift_
                 | (mu.trackSubAddress(RegionalMuonCand::kME4Order) & 0x1) << emtfTrAddrMe4OrderShift_;
    } else {
      edm::LogWarning("L1T") << "EMTF muon track address map contains " << mu.trackAddress().size() << " instead of the expected " << RegionalMuonCand::kNumEmtfSubAddr << " subaddresses. Check the data format. Setting track address to 0.";
      rawTrkAddr = 0;
    }
  } else {
    rawTrkAddr = mu.trackAddress().at(0);
  }

  raw_data_32_63 = mu.hwSign() << signShift_
                 | mu.hwSignValid() << signValidShift_
                 | (rawTrkAddr & trackAddressMask_) << trackAddressShift_;
}

uint64_t 
l1t::RegionalMuonRawDigiTranslator::generate64bitDataWord(const RegionalMuonCand& mu)
{
  uint32_t lsw;
  uint32_t msw;

  generatePackedDataWords(mu, lsw, msw);
  return (((uint64_t)msw) << 32) + lsw;
}

