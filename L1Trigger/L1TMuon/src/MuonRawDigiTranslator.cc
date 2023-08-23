#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"
#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

void l1t::MuonRawDigiTranslator::fillMuon(
    Muon& mu, uint32_t raw_data_spare, uint32_t raw_data_00_31, uint32_t raw_data_32_63, int fed, int fw, int muInBx) {
  // Need the hw charge to properly compute dPhi
  mu.setHwCharge((raw_data_32_63 >> chargeShift_) & 0x1);

  // The position of the eta and phi coordinates in the RAW data changed between the 2016 run and the 2017 run.
  // Eta and phi at the muon system are replaced by eta and phi at the vertex
  // Eta and phi at the muon system are moved to spare bits
  // In Run-3 we have displacement information.
  // To make room for these data the raw eta value was moved to the second "spare" word which we will have to treat separately
  // The uGMT (FED 1402) or uGT (FED 1404) FW versions are used to determine the era.
  if ((fed == kUgmtFedId && fw < kUgmtFwVersionUntil2016) || (fed == kUgtFedId && fw < kUgtFwVersionUntil2016)) {
    fillMuonCoordinates2016(mu, raw_data_00_31, raw_data_32_63);
  } else if ((fed == kUgmtFedId && fw < kUgmtFwVersionUntil2017) || (fed == kUgtFedId && fw < kUgtFwVersionUntil2017)) {
    fillMuonCoordinatesFrom2017(mu, raw_data_00_31, raw_data_32_63);
  } else if ((fed == kUgmtFedId && fw == kUgmtFwVersionRun3Start) ||
             (fed == kUgtFedId && fw < kUgtFwVersionUntilRun3Start)) {
    // We're unpacking data from the November MWGR where the raw eta values were shifted by one bit.
    fillMuonQuantitiesRun3(mu, raw_data_spare, raw_data_00_31, raw_data_32_63, muInBx, true);
  } else {
    fillMuonQuantitiesRun3(mu, raw_data_spare, raw_data_00_31, raw_data_32_63, muInBx, false);
  }

  // Fill pT, qual, iso, charge, index bits, coordinates at vtx
  fillMuonStableQuantities(mu, raw_data_00_31, raw_data_32_63);
}

void l1t::MuonRawDigiTranslator::fillIntermediateMuon(Muon& mu,
                                                      uint32_t raw_data_00_31,
                                                      uint32_t raw_data_32_63,
                                                      int fw) {
  // Need the hw charge to properly compute dPhi
  mu.setHwCharge((raw_data_32_63 >> chargeShift_) & 0x1);

  if (fw < kUgmtFwVersionUntil2016) {
    fillMuonCoordinates2016(mu, raw_data_00_31, raw_data_32_63);
  } else if (fw < kUgmtFwVersionUntil2017) {
    fillMuonCoordinatesFrom2017(mu, raw_data_00_31, raw_data_32_63);
  } else {
    fillIntermediateMuonQuantitiesRun3(mu, raw_data_00_31, raw_data_32_63);
  }

  // Fill pT, qual, iso, charge, index bits,  phys. coordinates at vtx & unconstrained pT
  fillMuonStableQuantities(mu, raw_data_00_31, raw_data_32_63);
}

void l1t::MuonRawDigiTranslator::fillMuonStableQuantities(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63) {
  mu.setHwPt((raw_data_00_31 >> ptShift_) & ptMask_);
  mu.setHwQual((raw_data_00_31 >> qualShift_) & qualMask_);
  mu.setHwIso((raw_data_32_63 >> isoShift_) & isoMask_);
  // charge is coded as -1^chargeBit
  mu.setHwChargeValid((raw_data_32_63 >> chargeValidShift_) & 0x1);
  mu.setTfMuonIndex((raw_data_32_63 >> tfMuonIndexShift_) & tfMuonIndexMask_);
  if (mu.hwChargeValid()) {
    mu.setCharge(1 - 2 * mu.hwCharge());
  } else {
    mu.setCharge(0);
  }

  math::PtEtaPhiMLorentzVector vec{(mu.hwPt() - 1) * 0.5, mu.hwEta() * 0.010875, mu.hwPhi() * 0.010908, 0.0};
  mu.setP4(vec);
  // generate a muon at the vertex to extract the physical eta and phi coordinates
  math::PtEtaPhiMLorentzVector vecAtVtx{
      (mu.hwPt() - 1) * 0.5, mu.hwEtaAtVtx() * 0.010875, mu.hwPhiAtVtx() * 0.010908, 0.0};
  Muon muAtVtx;
  muAtVtx.setP4(vecAtVtx);
  mu.setEtaAtVtx(muAtVtx.eta());
  mu.setPhiAtVtx(muAtVtx.phi());

  int hwPtUnconstrained{mu.hwPtUnconstrained()};
  mu.setPtUnconstrained(
      hwPtUnconstrained == 0 ? 0 : (hwPtUnconstrained - 1));  // Don't want negative pT, unconstr. pT has LSB of 1 GeV.
}

void l1t::MuonRawDigiTranslator::fillMuon(
    Muon& mu, uint32_t raw_data_spare, uint64_t dataword, int fed, int fw, int muInBx) {
  fillMuon(
      mu, raw_data_spare, (uint32_t)(dataword & 0xFFFFFFFF), (uint32_t)((dataword >> 32) & 0xFFFFFFFF), fed, fw, muInBx);
}

void l1t::MuonRawDigiTranslator::fillMuonCoordinates2016(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63) {
  // coordinates at the muon system are in 2016 where in 2017 eta and phi at the vertex are
  mu.setHwEta(calcHwEta(raw_data_00_31, absEtaAtVtxShift_, etaAtVtxSignShift_));
  mu.setHwPhi((raw_data_00_31 >> phiAtVtxShift_) & phiMask_);

  // set the coordiantes at vertex to be the same as the coordinates at the muon system
  mu.setHwEtaAtVtx(mu.hwEta());
  mu.setHwPhiAtVtx(mu.hwPhi());
  // deltas are 0
  mu.setHwDEtaExtra(0);
  mu.setHwDPhiExtra(0);
}

void l1t::MuonRawDigiTranslator::fillMuonCoordinatesFrom2017(Muon& mu,
                                                             uint32_t raw_data_00_31,
                                                             uint32_t raw_data_32_63) {
  // coordinates at the muon system
  mu.setHwEta(calcHwEta(raw_data_32_63, absEtaShift_, etaSignShift_));
  mu.setHwPhi((raw_data_32_63 >> phiShift_) & phiMask_);

  // coordinates at the vertex
  mu.setHwEtaAtVtx(calcHwEta(raw_data_00_31, absEtaAtVtxShift_, etaAtVtxSignShift_));
  mu.setHwPhiAtVtx((raw_data_00_31 >> phiAtVtxShift_) & phiMask_);
  // deltas
  mu.setHwDEtaExtra(mu.hwEtaAtVtx() - mu.hwEta());
  int dPhi = mu.hwPhiAtVtx() - mu.hwPhi();
  if (mu.hwCharge() == 1 && dPhi > 0) {
    dPhi -= 576;
  } else if (mu.hwCharge() == 0 && dPhi < 0) {
    dPhi += 576;
  }
  mu.setHwDPhiExtra(dPhi);
}

void l1t::MuonRawDigiTranslator::fillMuonQuantitiesRun3(Muon& mu,
                                                        uint32_t raw_data_spare,
                                                        uint32_t raw_data_00_31,
                                                        uint32_t raw_data_32_63,
                                                        int muInBx,
                                                        bool wasSpecialMWGR /*= false*/) {
  unsigned absEtaMu1Shift{absEtaMu1Shift_};
  unsigned etaMu1SignShift{etaMu1SignShift_};
  unsigned absEtaMu2Shift{absEtaMu2Shift_};
  unsigned etaMu2SignShift{etaMu2SignShift_};

  // Adjust if we're unpacking data from the November 2020 MWGR.
  if (wasSpecialMWGR) {
    --absEtaMu1Shift;
    --etaMu1SignShift;
    --absEtaMu2Shift;
    --etaMu2SignShift;
  }
  // coordinates at the muon system
  // Where to find the raw eta depends on which muon we're looking at
  if (muInBx == 1) {
    mu.setHwEta(calcHwEta(raw_data_spare, absEtaMu1Shift, etaMu1SignShift));
  } else if (muInBx == 2) {
    mu.setHwEta(calcHwEta(raw_data_spare, absEtaMu2Shift, etaMu2SignShift));
  } else {
    edm::LogWarning("L1T") << "Received invalid muon id " << muInBx << ". Cannot fill eta value in the muon system.";
  }
  mu.setHwPhi((raw_data_32_63 >> phiShift_) & phiMask_);

  // coordinates at the vertex
  mu.setHwEtaAtVtx(calcHwEta(raw_data_00_31, absEtaAtVtxShift_, etaAtVtxSignShift_));
  mu.setHwPhiAtVtx((raw_data_00_31 >> phiAtVtxShift_) & phiMask_);
  // deltas
  mu.setHwDEtaExtra(mu.hwEtaAtVtx() - mu.hwEta());
  int dPhi = mu.hwPhiAtVtx() - mu.hwPhi();
  if (mu.hwCharge() == 1 && dPhi > 0) {
    dPhi -= 576;
  } else if (mu.hwCharge() == 0 && dPhi < 0) {
    dPhi += 576;
  }
  mu.setHwDPhiExtra(dPhi);

  // displacement information
  mu.setHwDXY((raw_data_32_63 >> dxyShift_) & dxyMask_);
  mu.setHwPtUnconstrained((raw_data_32_63 >> ptUnconstrainedShift_) & ptUnconstrainedMask_);
}

void l1t::MuonRawDigiTranslator::fillIntermediateMuonQuantitiesRun3(Muon& mu,
                                                                    uint32_t raw_data_00_31,
                                                                    uint32_t raw_data_32_63) {
  fillMuonCoordinatesFrom2017(mu, raw_data_00_31, raw_data_32_63);

  // displacement information
  mu.setHwDXY((raw_data_32_63 >> dxyShift_) & dxyMask_);
  mu.setHwPtUnconstrained((raw_data_00_31 >> ptUnconstrainedIntermedidateShift_) & ptUnconstrainedMask_);
}

void l1t::MuonRawDigiTranslator::generatePackedMuonDataWords(const Muon& mu,
                                                             uint32_t& raw_data_spare,
                                                             uint32_t& raw_data_00_31,
                                                             uint32_t& raw_data_32_63,
                                                             const int fedID,
                                                             const int fwID,
                                                             const int muInBx) {
  int abs_eta = mu.hwEta();
  if (abs_eta < 0) {
    abs_eta += (1 << (etaSignShift_ - absEtaShift_));
  }
  int abs_eta_at_vtx = mu.hwEtaAtVtx();
  if (abs_eta_at_vtx < 0) {
    abs_eta_at_vtx += (1 << (etaAtVtxSignShift_ - absEtaAtVtxShift_));
  }
  if ((fedID == kUgmtFedId && fwID < kUgmtFwVersionUntil2016) ||
      (fedID == kUgtFedId && fwID < kUgtFwVersionUntil2016)) {
    // For 2016 the non-extrapolated coordiantes were in the place that are now occupied by the extrapolated quantities.
    raw_data_spare = 0;
    raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_ | (mu.hwQual() & qualMask_) << qualShift_ |
                     (abs_eta & absEtaMask_) << absEtaAtVtxShift_ | (mu.hwEta() < 0) << etaAtVtxSignShift_ |
                     (mu.hwPhi() & phiMask_) << phiAtVtxShift_;
    raw_data_32_63 = mu.hwCharge() << chargeShift_ | mu.hwChargeValid() << chargeValidShift_ |
                     (mu.tfMuonIndex() & tfMuonIndexMask_) << tfMuonIndexShift_ | (mu.hwIso() & isoMask_) << isoShift_;
  } else if ((fedID == kUgmtFedId && fwID < kUgmtFwVersionUntil2017) ||
             (fedID == kUgtFedId && fwID < kUgtFwVersionUntil2017)) {
    raw_data_spare = 0;
    raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_ | (mu.hwQual() & qualMask_) << qualShift_ |
                     (abs_eta_at_vtx & absEtaMask_) << absEtaAtVtxShift_ | (mu.hwEtaAtVtx() < 0) << etaAtVtxSignShift_ |
                     (mu.hwPhiAtVtx() & phiMask_) << phiAtVtxShift_;

    raw_data_32_63 = mu.hwCharge() << chargeShift_ | mu.hwChargeValid() << chargeValidShift_ |
                     (mu.tfMuonIndex() & tfMuonIndexMask_) << tfMuonIndexShift_ | (mu.hwIso() & isoMask_) << isoShift_ |
                     (abs_eta & absEtaMask_) << absEtaShift_ | (mu.hwEta() < 0) << etaSignShift_ |
                     (mu.hwPhi() & phiMask_) << phiShift_;
  } else if ((fedID == kUgmtFedId && fwID == kUgmtFwVersionRun3Start) ||
             (fedID == kUgtFedId &&
              fwID < kUgtFwVersionUntilRun3Start)) {  // This allows us to unpack data taken in the November 2020 MWGR.
    generatePackedMuonDataWordsRun3(
        mu, abs_eta, abs_eta_at_vtx, raw_data_spare, raw_data_00_31, raw_data_32_63, muInBx, true);
  } else {
    generatePackedMuonDataWordsRun3(
        mu, abs_eta, abs_eta_at_vtx, raw_data_spare, raw_data_00_31, raw_data_32_63, muInBx, false);
  }
}

void l1t::MuonRawDigiTranslator::generatePackedMuonDataWordsRun3(const Muon& mu,
                                                                 const int abs_eta,
                                                                 const int abs_eta_at_vtx,
                                                                 uint32_t& raw_data_spare,
                                                                 uint32_t& raw_data_00_31,
                                                                 uint32_t& raw_data_32_63,
                                                                 const int muInBx,
                                                                 const bool wasSpecialMWGR /*= false*/) {
  int absEtaShiftRun3{0}, etaSignShiftRun3{0};
  if (muInBx == 1) {
    absEtaShiftRun3 = absEtaMu1Shift_;
    etaSignShiftRun3 = etaMu1SignShift_;
  } else if (muInBx == 2) {
    absEtaShiftRun3 = absEtaMu2Shift_;
    etaSignShiftRun3 = etaMu2SignShift_;
  }

  // Adjust if we're packing the November 2020 MWGR
  if (wasSpecialMWGR && (muInBx == 1 || muInBx == 2)) {
    --absEtaShiftRun3;
    --etaSignShiftRun3;
  }

  raw_data_spare = (abs_eta & absEtaMask_) << absEtaShiftRun3 | (mu.hwEta() < 0) << etaSignShiftRun3;
  raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_ | (mu.hwQual() & qualMask_) << qualShift_ |
                   (abs_eta_at_vtx & absEtaMask_) << absEtaAtVtxShift_ | (mu.hwEtaAtVtx() < 0) << etaAtVtxSignShift_ |
                   (mu.hwPhiAtVtx() & phiMask_) << phiAtVtxShift_;
  raw_data_32_63 = mu.hwCharge() << chargeShift_ | mu.hwChargeValid() << chargeValidShift_ |
                   (mu.tfMuonIndex() & tfMuonIndexMask_) << tfMuonIndexShift_ | (mu.hwIso() & isoMask_) << isoShift_ |
                   (mu.hwPhi() & phiMask_) << phiShift_ |
                   (mu.hwPtUnconstrained() & ptUnconstrainedMask_) << ptUnconstrainedShift_ |
                   (mu.hwDXY() & dxyMask_) << dxyShift_;
}

void l1t::MuonRawDigiTranslator::generate64bitDataWord(
    const Muon& mu, uint32_t& raw_data_spare, uint64_t& dataword, int fedId, int fwId, int muInBx) {
  uint32_t lsw;
  uint32_t msw;

  generatePackedMuonDataWords(mu, raw_data_spare, lsw, msw, fedId, fwId, muInBx);
  dataword = (((uint64_t)msw) << 32) + lsw;
}

bool l1t::MuonRawDigiTranslator::showerFired(uint32_t shower_word, int fedId, int fwId) {
  if ((fedId == kUgmtFedId && fwId >= kUgmtFwVersionFirstWithShowers) ||
      (fedId == kUgtFedId && fwId >= kUgtFwVersionFirstWithShowers)) {
    return ((shower_word >> showerShift_) & 1) == 1;
  }
  return false;
}

std::array<std::array<uint32_t, 4>, 2> l1t::MuonRawDigiTranslator::getPackedShowerDataWords(const MuonShower& shower,
                                                                                            const int fedId,
                                                                                            const int fwId) {
  std::array<std::array<uint32_t, 4>, 2> res{};
  if ((fedId == kUgmtFedId && fwId >= kUgmtFwVersionFirstWithShowers) ||
      (fedId == kUgtFedId && fwId >= kUgtFwVersionFirstWithShowers)) {
    res.at(0).at(0) = shower.isOneNominalInTime() ? (1 << showerShift_) : 0;
    res.at(0).at(1) = shower.isOneTightInTime() ? (1 << showerShift_) : 0;
  }
  if ((fedId == kUgmtFedId && fwId >= kUgmtFwVersionShowersFrom2023_patched) ||
      (fedId == kUgtFedId && fwId >= kUgtFwVersionShowersFrom2023)) {
    res.at(1).at(1) = shower.isTwoLooseDiffSectorsInTime()
                          ? (1 << showerShift_)
                          : 0;  // The two loose shower is on the second link in the second muon word.
  } else if (fedId == kUgmtFedId && fwId >= kUgmtFwVersionShowersFrom2023) {
    res.at(1).at(0) = shower.isTwoLooseDiffSectorsInTime()
                          ? (1 << showerShift_)
                          : 0;  // uGMT was briefly transmitting it on the first link instead.
  }
  return res;
}

int l1t::MuonRawDigiTranslator::calcHwEta(const uint32_t& raw,
                                          const unsigned absEtaShift,
                                          const unsigned etaSignShift) {
  // eta is coded as two's complement
  int abs_eta = (raw >> absEtaShift) & absEtaMask_;
  if ((raw >> etaSignShift) & 0x1) {
    return abs_eta - (1 << (etaSignShift - absEtaShift));
  } else {
    return abs_eta;
  }
}
