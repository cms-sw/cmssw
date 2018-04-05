#include "TMath.h"
#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

void
l1t::MuonRawDigiTranslator::fillMuon(Muon& mu, uint32_t raw_data_00_31, uint32_t raw_data_32_63, int fed, unsigned int fw)
{
  int hwPt = (raw_data_00_31 >> ptShift_) & ptMask_;
  if (hwPt > 0) {
    mu.setHwPt(hwPt);
    mu.setHwQual((raw_data_00_31 >> qualShift_) & qualMask_);
    mu.setHwIso((raw_data_32_63 >> isoShift_) & isoMask_);
    // charge is coded as -1^chargeBit
    mu.setHwCharge((raw_data_32_63 >> chargeShift_) & 0x1);
    mu.setHwChargeValid((raw_data_32_63 >> chargeValidShift_) & 0x1);
    mu.setTfMuonIndex((raw_data_32_63 >> tfMuonIndexShift_) & tfMuonIndexMask_);

    // The position of the eta and phi coordinates in the RAW data changed between the 2016 run and the 2017 run.
    // Eta and phi at the muon system are replaced by eta and phi at the vertex
    // Eta and phi at the muon system are moved to spare bits
    // The uGMT (FED 1402) or uGT (FED 1404) FW versions are used to determine the era.
    if ((fed == 1402 && fw < 0x4010000) || (fed == 1404 && fw < 0x10A6)) {
      // coordinates at the muon system are in 2016 where in 2017 eta and phi at the vertex are
      mu.setHwEta(calcHwEta(raw_data_00_31, absEtaAtVtxShift_, etaAtVtxSignShift_));
      mu.setHwPhi((raw_data_00_31 >> phiAtVtxShift_) & phiMask_);

      // set the coordiantes at vertex to be the same as the coordinates at the muon system
      mu.setHwEtaAtVtx(mu.hwEta());
      mu.setHwPhiAtVtx(mu.hwPhi());
      // deltas are 0
      mu.setHwDEtaExtra(0);
      mu.setHwDPhiExtra(0);
    } else {
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

    math::PtEtaPhiMLorentzVector vec{(mu.hwPt()-1)*0.5, mu.hwEta()*0.010875, mu.hwPhi()*0.010908, 0.0};
    mu.setP4(vec);
    // generate a muon at the vertex to extract the physical eta and phi coordinates
    math::PtEtaPhiMLorentzVector vecAtVtx{(mu.hwPt()-1)*0.5, mu.hwEtaAtVtx()*0.010875, mu.hwPhiAtVtx()*0.010908, 0.0};
    Muon muAtVtx;
    muAtVtx.setP4(vecAtVtx);
    mu.setEtaAtVtx(muAtVtx.eta());
    mu.setPhiAtVtx(muAtVtx.phi());
    if (mu.hwChargeValid()) {
      mu.setCharge(1 - 2 * mu.hwCharge());
    } else {
      mu.setCharge(0);
    }
  }
}

void
l1t::MuonRawDigiTranslator::fillMuon(Muon& mu, uint64_t dataword, int fed, unsigned int fw)
{
  fillMuon(mu, (uint32_t)(dataword & 0xFFFFFFFF), (uint32_t)((dataword >> 32) & 0xFFFFFFFF), fed, fw);
}

void
l1t::MuonRawDigiTranslator::generatePackedDataWords(const Muon& mu, uint32_t &raw_data_00_31, uint32_t &raw_data_32_63)
{
  int abs_eta = mu.hwEta();
  if (abs_eta < 0) {
    abs_eta += (1 << (etaSignShift_ - absEtaShift_));
  }
  int abs_eta_at_vtx = mu.hwEtaAtVtx();
  if (abs_eta_at_vtx < 0) {
    abs_eta_at_vtx += (1 << (etaAtVtxSignShift_ - absEtaAtVtxShift_));
  }
  raw_data_00_31 = (mu.hwPt() & ptMask_) << ptShift_
                 | (mu.hwQual() & qualMask_) << qualShift_
                 | (abs_eta_at_vtx & absEtaMask_) << absEtaAtVtxShift_
                 | (mu.hwEtaAtVtx() < 0) << etaAtVtxSignShift_
                 | (mu.hwPhiAtVtx() & phiMask_) << phiAtVtxShift_;

  raw_data_32_63 = mu.hwCharge() << chargeShift_
                 | mu.hwChargeValid() << chargeValidShift_
                 | (mu.tfMuonIndex() & tfMuonIndexMask_) << tfMuonIndexShift_
                 | (mu.hwIso() & isoMask_) << isoShift_
                 | (abs_eta & absEtaMask_) << absEtaShift_
                 | (mu.hwEta() < 0) << etaSignShift_
                 | (mu.hwPhi() & phiMask_) << phiShift_;
}

uint64_t 
l1t::MuonRawDigiTranslator::generate64bitDataWord(const Muon& mu)
{
  uint32_t lsw;
  uint32_t msw;

  generatePackedDataWords(mu, lsw, msw);
  return (((uint64_t)msw) << 32) + lsw;
}

int
l1t::MuonRawDigiTranslator::calcHwEta(const uint32_t& raw, const unsigned absEtaShift, const unsigned etaSignShift)
{
  // eta is coded as two's complement
  int abs_eta = (raw >> absEtaShift) & absEtaMask_;
  if ((raw >> etaSignShift) & 0x1) {
     return abs_eta - (1 << (etaSignShift - absEtaShift));
  } else {
     return abs_eta;
  }
}

