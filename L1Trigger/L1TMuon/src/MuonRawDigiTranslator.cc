#include "TMath.h"
#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

void
l1t::MuonRawDigiTranslator::fillMuon(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63)
{
  mu.setHwPt((raw_data_00_31 >> ptShift_) & ptMask_);
  mu.setHwQual((raw_data_00_31 >> qualShift_) & qualMask_);
  
  // eta is coded as two's complement
  int abs_eta = (raw_data_00_31 >> absEtaShift_) & absEtaMask_;
  if ((raw_data_00_31 >> etaSignShift_) & 0x1) {
     mu.setHwEta(abs_eta - (1 << (etaSignShift_ - absEtaShift_)));
  } else {
     mu.setHwEta(abs_eta);
  }

  mu.setHwPhi((raw_data_00_31 >> phiShift_) & phiMask_);
  mu.setHwIso((raw_data_32_63 >> isoShift_) & isoMask_); 
  // charge is coded as -1^chargeBit
  mu.setHwCharge((raw_data_32_63 >> chargeShift_) & 0x1);
  mu.setHwChargeValid((raw_data_32_63 >> chargeValidShift_) & 0x1);
  mu.setTfMuonIndex((raw_data_32_63 >> tfMuonIndexShift_) & tfMuonIndexMask_);

  if (mu.hwPt() > 0) {
    math::PtEtaPhiMLorentzVector vec{(mu.hwPt()-1)*0.5, mu.hwEta()*0.010875, mu.hwPhi()*0.010908, 0.0};
    mu.setP4(vec);
    if (mu.hwChargeValid()) {
      mu.setCharge(1 - 2 * mu.hwCharge());
    } else {
      mu.setCharge(0);
    }
  }
}

void
l1t::MuonRawDigiTranslator::fillMuon(Muon& mu, uint64_t dataword)
{
  fillMuon(mu, (uint32_t)(dataword & 0xFFFFFFFF), (uint32_t)((dataword >> 32) & 0xFFFFFFFF));
}

void
l1t::MuonRawDigiTranslator::generatePackedDataWords(const Muon& mu, uint32_t &raw_data_00_31, uint32_t &raw_data_32_63)
{
  int abs_eta = mu.hwEta();
  if (abs_eta < 0) {
    abs_eta += (1 << (etaSignShift_ - absEtaShift_));
  }
  raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_
                 | (mu.hwQual() & qualMask_) << qualShift_
                 | (abs_eta & absEtaMask_) << absEtaShift_
                 | (mu.hwEta() < 0) << etaSignShift_
                 | (mu.hwPhi() & phiMask_) << phiShift_;

  raw_data_32_63 = mu.hwCharge() << chargeShift_
                 | mu.hwChargeValid() << chargeValidShift_
                 | (mu.tfMuonIndex() & tfMuonIndexMask_) << tfMuonIndexShift_
                 | (mu.hwIso() & isoMask_) << isoShift_;
}

uint64_t 
l1t::MuonRawDigiTranslator::generate64bitDataWord(const Muon& mu)
{
  uint32_t lsw;
  uint32_t msw;

  generatePackedDataWords(mu, lsw, msw);
  return (((uint64_t)msw) << 32) + lsw;
}

