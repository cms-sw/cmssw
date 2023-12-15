#include "L1TriggerScouting/Utilities/interface/convertToL1TFormat.h"

namespace l1ScoutingRun3 {

  l1t::Muon getL1TMuon(const Muon& muon) {
    return l1t::Muon(
        math::PtEtaPhiMLorentzVector(ugmt::fPt(muon.hwPt()), ugmt::fEta(muon.hwEta()), ugmt::fPhi(muon.hwPhi()), 0.),
        muon.hwPt(),
        muon.hwEta(),
        muon.hwPhi(),
        muon.hwQual(),
        muon.hwCharge(),
        muon.hwChargeValid(),
        0,
        muon.tfMuonIndex(),
        0,
        false,
        0,
        0,
        0,
        0,
        muon.hwEtaAtVtx(),
        muon.hwPhiAtVtx(),
        ugmt::fEtaAtVtx(muon.hwEtaAtVtx()),
        ugmt::fPhiAtVtx(muon.hwPhiAtVtx()),
        muon.hwPtUnconstrained(),
        ugmt::fPtUnconstrained(muon.hwPtUnconstrained()),
        muon.hwDXY());
  }

  l1t::Jet getL1TJet(const Jet& jet) {
    return l1t::Jet(
        math::PtEtaPhiMLorentzVector(demux::fEt(jet.hwEt()), demux::fEta(jet.hwEta()), demux::fPhi(jet.hwPhi()), 0.),
        jet.hwEt(),
        jet.hwEta(),
        jet.hwPhi(),
        jet.hwIso());
  }

  l1t::EGamma getL1TEGamma(const EGamma& eGamma) {
    return l1t::EGamma(math::PtEtaPhiMLorentzVector(
                           demux::fEt(eGamma.hwEt()), demux::fEta(eGamma.hwEta()), demux::fPhi(eGamma.hwPhi()), 0.),
                       eGamma.hwEt(),
                       eGamma.hwEta(),
                       eGamma.hwPhi(),
                       0,
                       eGamma.hwIso());
  }

  l1t::Tau getL1TTau(const Tau& tau) {
    return l1t::Tau(
        math::PtEtaPhiMLorentzVector(demux::fEt(tau.hwEt()), demux::fEta(tau.hwEta()), demux::fPhi(tau.hwPhi()), 0.),
        tau.hwEt(),
        tau.hwEta(),
        tau.hwPhi(),
        0,
        tau.hwIso());
  }

  l1t::EtSum getL1TEtSum(const BxSums& sums, l1t::EtSum::EtSumType sumType) {
    switch (sumType) {
      case l1t::EtSum::kTotalEt:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwTotalEt()), 0., 0., 0.), sumType, sums.hwTotalEt(), 0, 0, 0);
      case l1t::EtSum::kTotalEtEm:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwTotalEtEm()), 0., 0., 0.),
                          sumType,
                          sums.hwTotalEtEm(),
                          0,
                          0,
                          0);
      case l1t::EtSum::kTotalHt:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwTotalHt()), 0., 0., 0.), sumType, sums.hwTotalHt(), 0, 0, 0);
      case l1t::EtSum::kMissingEt:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwMissEt()), 0., demux::fPhi(sums.hwMissEtPhi()), 0.),
            sumType,
            sums.hwMissEt(),
            0,
            sums.hwMissEtPhi(),
            0);
      case l1t::EtSum::kMissingHt:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwMissHt()), 0., demux::fPhi(sums.hwMissHtPhi()), 0.),
            sumType,
            sums.hwMissHt(),
            0,
            sums.hwMissHtPhi(),
            0);
      case l1t::EtSum::kMissingEtHF:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwMissEtHF()), 0., demux::fPhi(sums.hwMissEtHFPhi()), 0.),
            sumType,
            sums.hwMissEtHF(),
            0,
            sums.hwMissEtHFPhi(),
            0);
      case l1t::EtSum::kMissingHtHF:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwMissHtHF()), 0., demux::fPhi(sums.hwMissHtHFPhi()), 0.),
            sumType,
            sums.hwMissHtHF(),
            0,
            sums.hwMissHtHFPhi(),
            0);
      case l1t::EtSum::kAsymEt:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwAsymEt()), 0., 0., 0.), sumType, sums.hwAsymEt(), 0, 0, 0);
      case l1t::EtSum::kAsymHt:
        return l1t::EtSum(
            math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwAsymHt()), 0., 0., 0.), sumType, sums.hwAsymHt(), 0, 0, 0);
      case l1t::EtSum::kAsymEtHF:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwAsymEtHF()), 0., 0., 0.),
                          sumType,
                          sums.hwAsymEtHF(),
                          0,
                          0,
                          0);
      case l1t::EtSum::kAsymHtHF:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(demux::fEt(sums.hwAsymHtHF()), 0., 0., 0.),
                          sumType,
                          sums.hwAsymHtHF(),
                          0,
                          0,
                          0);
      case l1t::EtSum::kMinBiasHFP0:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), sumType, sums.minBiasHFP0(), 0, 0, 0);
      case l1t::EtSum::kMinBiasHFP1:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), sumType, sums.minBiasHFP1(), 0, 0, 0);
      case l1t::EtSum::kMinBiasHFM0:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), sumType, sums.minBiasHFM0(), 0, 0, 0);
      case l1t::EtSum::kMinBiasHFM1:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), sumType, sums.minBiasHFM1(), 0, 0, 0);
      case l1t::EtSum::kCentrality:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), sumType, sums.centrality(), 0, 0, 0);
      case l1t::EtSum::kTowerCount:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), sumType, sums.towerCount(), 0, 0, 0);
      default:
        return l1t::EtSum(math::PtEtaPhiMLorentzVector(0., 0., 0., 0.), l1t::EtSum::kUninitialized, 0, 0, 0, 0);
    }
  }

}  // namespace l1ScoutingRun3