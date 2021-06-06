#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"

void l1t::RegionalMuonRawDigiTranslator::fillRegionalMuonCand(RegionalMuonCand& mu,
                                                              const uint32_t raw_data_00_31,
                                                              const uint32_t raw_data_32_63,
                                                              const int proc,
                                                              const tftype tf,
                                                              const bool isRun3) {
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
    int detSide = (rawTrackAddress >> bmtfTrAddrDetSideShift_) & 0x1;
    int wheelNum = (rawTrackAddress >> bmtfTrAddrWheelShift_) & bmtfTrAddrWheelMask_;
    int statAddr1 = ((rawTrackAddress >> bmtfTrAddrStat1Shift_) & bmtfTrAddrStat1Mask_);
    int statAddr2 = ((rawTrackAddress >> bmtfTrAddrStat2Shift_) & bmtfTrAddrStat2Mask_);
    int statAddr3 = ((rawTrackAddress >> bmtfTrAddrStat3Shift_) & bmtfTrAddrStat3Mask_);
    int statAddr4 = ((rawTrackAddress >> bmtfTrAddrStat4Shift_) & bmtfTrAddrStat4Mask_);

    mu.setTrackSubAddress(RegionalMuonCand::kWheelSide, detSide);
    mu.setTrackSubAddress(RegionalMuonCand::kWheelNum, wheelNum);
    if (!isRun3) {  // The Run-2 standard configuration
      mu.setTrackSubAddress(RegionalMuonCand::kStat1, statAddr1);
      mu.setTrackSubAddress(RegionalMuonCand::kStat2, statAddr2);
      mu.setTrackSubAddress(RegionalMuonCand::kStat3, statAddr3);
      mu.setTrackSubAddress(RegionalMuonCand::kStat4, statAddr4);
    } else {
      // For Run-3 track address encoding has changed as the Kalman Filter tracks from outside in.
      // As a result station assignment is inverted
      // (i.e. the field that contained the station 1 information for Run-2 now contains station 4 information and so on.)
      mu.setTrackSubAddress(RegionalMuonCand::kStat1, statAddr4);
      mu.setTrackSubAddress(RegionalMuonCand::kStat2, statAddr3);
      mu.setTrackSubAddress(RegionalMuonCand::kStat3, statAddr2);
      mu.setTrackSubAddress(RegionalMuonCand::kStat4, statAddr1);
      // Additionally we now have displacement information from the BMTF
      mu.setHwPtUnconstrained((raw_data_32_63 >> bmtfPtUnconstrainedShift_) & ptUnconstrainedMask_);
      mu.setHwDXY((raw_data_32_63 >> bmtfDxyShift_) & dxyMask_);
    }
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat1, 0);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat2, 0);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat3, 0);
    mu.setTrackSubAddress(RegionalMuonCand::kSegSelStat4, 0);
    //mu.setTrackSubAddress(RegionalMuonCand::kNumBmtfSubAddr, 0);
  } else if (tf == emtf_neg || tf == emtf_pos) {
    mu.setTrackSubAddress(RegionalMuonCand::kME1Seg, (rawTrackAddress >> emtfTrAddrMe1SegShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME1Ch, (rawTrackAddress >> emtfTrAddrMe1ChShift_) & emtfTrAddrMe1ChMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME2Seg, (rawTrackAddress >> emtfTrAddrMe2SegShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME2Ch, (rawTrackAddress >> emtfTrAddrMe2ChShift_) & emtfTrAddrMe2ChMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME3Seg, (rawTrackAddress >> emtfTrAddrMe3SegShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME3Ch, (rawTrackAddress >> emtfTrAddrMe3ChShift_) & emtfTrAddrMe3ChMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kME4Seg, (rawTrackAddress >> emtfTrAddrMe4SegShift_) & 0x1);
    mu.setTrackSubAddress(RegionalMuonCand::kME4Ch, (rawTrackAddress >> emtfTrAddrMe4ChShift_) & emtfTrAddrMe4ChMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kTrkNum,
                          (rawTrackAddress >> emtfTrAddrTrkNumShift_) & emtfTrAddrTrkNumMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kBX, (rawTrackAddress >> emtfTrAddrBxShift_) & emtfTrAddrBxMask_);
    if (isRun3) {  // In Run-3 we receive displaced muon information from EMTF
      mu.setHwPtUnconstrained((raw_data_32_63 >> emtfPtUnconstrainedShift_) & ptUnconstrainedMask_);
      mu.setHwDXY((raw_data_32_63 >> emtfDxyShift_) & dxyMask_);
    }
  } else if (tf == omtf_neg || tf == omtf_pos) {
    mu.setTrackSubAddress(RegionalMuonCand::kLayers,
                          (rawTrackAddress >> omtfTrAddrLayersShift_) & omtfTrAddrLayersMask_);
    mu.setTrackSubAddress(RegionalMuonCand::kZero, 0);
    mu.setTrackSubAddress(RegionalMuonCand::kWeight,
                          (rawTrackAddress >> omtfTrAddrWeightShift_) & omtfTrAddrWeightMask_);
  } else {
    std::map<int, int> trackAddr;
    trackAddr[0] = rawTrackAddress;
    mu.setTrackAddress(trackAddr);
  }

  mu.setTFIdentifiers(proc, tf);
  mu.setDataword(raw_data_32_63, raw_data_00_31);
}

void l1t::RegionalMuonRawDigiTranslator::fillRegionalMuonCand(
    RegionalMuonCand& mu, const uint64_t dataword, const int proc, const tftype tf, const bool isRun3) {
  fillRegionalMuonCand(
      mu, (uint32_t)(dataword & 0xFFFFFFFF), (uint32_t)((dataword >> 32) & 0xFFFFFFFF), proc, tf, isRun3);
}

void l1t::RegionalMuonRawDigiTranslator::generatePackedDataWords(const RegionalMuonCand& mu,
                                                                 uint32_t& raw_data_00_31,
                                                                 uint32_t& raw_data_32_63,
                                                                 const bool isRun3) {
  int abs_eta = mu.hwEta();
  if (abs_eta < 0) {
    abs_eta += (1 << (etaSignShift_ - absEtaShift_));
  }
  int abs_phi = mu.hwPhi();
  if (abs_phi < 0) {
    abs_phi += (1 << (phiSignShift_ - absPhiShift_));
  }
  raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_ | (mu.hwQual() & qualMask_) << qualShift_ |
                   (abs_eta & absEtaMask_) << absEtaShift_ | (mu.hwEta() < 0) << etaSignShift_ |
                   (mu.hwHF() & hfMask_) << hfShift_ | (abs_phi & absPhiMask_) << absPhiShift_ |
                   (mu.hwPhi() < 0) << phiSignShift_;

  // generate the raw track address from the subaddresses
  int rawTrkAddr = generateRawTrkAddress(mu, isRun3);

  raw_data_32_63 = mu.hwSign() << signShift_ | mu.hwSignValid() << signValidShift_ |
                   (rawTrkAddr & trackAddressMask_) << trackAddressShift_;
  if (isRun3 && mu.trackFinderType() == bmtf) {
    raw_data_32_63 |= (mu.hwPtUnconstrained() & ptUnconstrainedMask_) << bmtfPtUnconstrainedShift_ |
                      (mu.hwDXY() & dxyMask_) << bmtfDxyShift_;
  } else if (isRun3 && (mu.trackFinderType() == emtf_pos || mu.trackFinderType() == emtf_neg)) {
    raw_data_32_63 |= (mu.hwPtUnconstrained() & ptUnconstrainedMask_) << emtfPtUnconstrainedShift_ |
                      (mu.hwDXY() & dxyMask_) << emtfDxyShift_;
  }
}

uint64_t l1t::RegionalMuonRawDigiTranslator::generate64bitDataWord(const RegionalMuonCand& mu, const bool isRun3) {
  uint32_t lsw;
  uint32_t msw;

  generatePackedDataWords(mu, lsw, msw, isRun3);
  return (((uint64_t)msw) << 32) + lsw;
}

int l1t::RegionalMuonRawDigiTranslator::generateRawTrkAddress(const RegionalMuonCand& mu, const bool isKalman) {
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
      if (isKalman) {
        stat1 = mu.trackSubAddress(RegionalMuonCand::kStat4);
        stat2 = mu.trackSubAddress(RegionalMuonCand::kStat3);
        stat3 = mu.trackSubAddress(RegionalMuonCand::kStat2);
        stat4 = mu.trackSubAddress(RegionalMuonCand::kStat1);
      }

      rawTrkAddr = (detSide & 0x1) << bmtfTrAddrDetSideShift_ |
                   (wheelNum & bmtfTrAddrWheelMask_) << bmtfTrAddrWheelShift_ |
                   (stat1 & bmtfTrAddrStat1Mask_) << bmtfTrAddrStat1Shift_ |
                   (stat2 & bmtfTrAddrStat2Mask_) << bmtfTrAddrStat2Shift_ |
                   (stat3 & bmtfTrAddrStat3Mask_) << bmtfTrAddrStat3Shift_ |
                   (stat4 & bmtfTrAddrStat4Mask_) << bmtfTrAddrStat4Shift_;
    } else {
      edm::LogWarning("L1T") << "BMTF muon track address map contains " << mu.trackAddress().size()
                             << " instead of the expected " << RegionalMuonCand::kNumBmtfSubAddr
                             << " subaddresses. Check the data format. Setting track address to 0.";
      rawTrkAddr = 0;
    }
  } else if (tf == emtf_neg || tf == emtf_pos) {
    // protection against a track address map with the wrong size
    if (mu.trackAddress().size() == RegionalMuonCand::kNumEmtfSubAddr) {
      rawTrkAddr = (mu.trackSubAddress(RegionalMuonCand::kME1Seg) & 0x1) << emtfTrAddrMe1SegShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME1Ch) & emtfTrAddrMe1ChMask_) << emtfTrAddrMe1ChShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME2Seg) & 0x1) << emtfTrAddrMe2SegShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME2Ch) & emtfTrAddrMe2ChMask_) << emtfTrAddrMe2ChShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME3Seg) & 0x1) << emtfTrAddrMe3SegShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME3Ch) & emtfTrAddrMe3ChMask_) << emtfTrAddrMe3ChShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME4Seg) & 0x1) << emtfTrAddrMe4SegShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kME4Ch) & emtfTrAddrMe4ChMask_) << emtfTrAddrMe4ChShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kTrkNum) & emtfTrAddrTrkNumMask_) << emtfTrAddrTrkNumShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kBX) & emtfTrAddrBxMask_) << emtfTrAddrBxShift_;

    } else {
      edm::LogWarning("L1T") << "EMTF muon track address map contains " << mu.trackAddress().size()
                             << " instead of the expected " << RegionalMuonCand::kNumEmtfSubAddr
                             << " subaddresses. Check the data format. Setting track address to 0.";
      rawTrkAddr = 0;
    }
  } else if (tf == omtf_neg || tf == omtf_pos) {
    // protection against a track address map with the wrong size
    if (mu.trackAddress().size() == RegionalMuonCand::kNumOmtfSubAddr) {
      rawTrkAddr = (mu.trackSubAddress(RegionalMuonCand::kLayers) & omtfTrAddrLayersMask_) << omtfTrAddrLayersShift_ |
                   (mu.trackSubAddress(RegionalMuonCand::kWeight) & omtfTrAddrWeightMask_) << omtfTrAddrWeightShift_;

    } else {
      edm::LogWarning("L1T") << "OMTF muon track address map contains " << mu.trackAddress().size()
                             << " instead of the expected " << RegionalMuonCand::kNumOmtfSubAddr
                             << " subaddresses. Check the data format. Setting track address to 0.";
      rawTrkAddr = 0;
    }
  } else {
    rawTrkAddr = mu.trackAddress().at(0);
  }

  return rawTrkAddr;
}
