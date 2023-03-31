#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Upgrade.h"

L1Analysis::L1AnalysisL1Upgrade::L1AnalysisL1Upgrade() {}

L1Analysis::L1AnalysisL1Upgrade::~L1AnalysisL1Upgrade() {}

void L1Analysis::L1AnalysisL1Upgrade::SetEm(const l1t::EGammaBxCollection& em, unsigned maxL1Upgrade) {
  for (int ibx = em.getFirstBX(); ibx <= em.getLastBX(); ++ibx) {
    for (l1t::EGammaBxCollection::const_iterator it = em.begin(ibx);
         it != em.end(ibx) && l1upgrade_.nEGs < maxL1Upgrade;
         it++) {
      if (it->pt() > 0) {
        l1upgrade_.egEt.push_back(it->pt());
        l1upgrade_.egEta.push_back(it->eta());
        l1upgrade_.egPhi.push_back(it->phi());
        l1upgrade_.egIEt.push_back(it->hwPt());
        l1upgrade_.egIEta.push_back(it->hwEta());
        l1upgrade_.egIPhi.push_back(it->hwPhi());
        l1upgrade_.egIso.push_back(it->hwIso());
        l1upgrade_.egBx.push_back(ibx);
        l1upgrade_.egTowerIPhi.push_back(it->towerIPhi());
        l1upgrade_.egTowerIEta.push_back(it->towerIEta());
        l1upgrade_.egRawEt.push_back(it->rawEt());
        l1upgrade_.egIsoEt.push_back(it->isoEt());
        l1upgrade_.egFootprintEt.push_back(it->footprintEt());
        l1upgrade_.egNTT.push_back(it->nTT());
        l1upgrade_.egShape.push_back(it->shape());
        l1upgrade_.egTowerHoE.push_back(it->towerHoE());
        l1upgrade_.egHwQual.push_back(it->hwQual());
        l1upgrade_.nEGs++;
      }
    }
  }
}

void L1Analysis::L1AnalysisL1Upgrade::SetTau(const l1t::TauBxCollection& tau, unsigned maxL1Upgrade) {
  for (int ibx = tau.getFirstBX(); ibx <= tau.getLastBX(); ++ibx) {
    for (l1t::TauBxCollection::const_iterator it = tau.begin(ibx);
         it != tau.end(ibx) && l1upgrade_.nTaus < maxL1Upgrade;
         it++) {
      if (it->pt() > 0) {
        l1upgrade_.tauEt.push_back(it->et());
        l1upgrade_.tauEta.push_back(it->eta());
        l1upgrade_.tauPhi.push_back(it->phi());
        l1upgrade_.tauIEt.push_back(it->hwPt());
        l1upgrade_.tauIEta.push_back(it->hwEta());
        l1upgrade_.tauIPhi.push_back(it->hwPhi());
        l1upgrade_.tauIso.push_back(it->hwIso());
        l1upgrade_.tauBx.push_back(ibx);
        l1upgrade_.tauTowerIPhi.push_back(it->towerIPhi());
        l1upgrade_.tauTowerIEta.push_back(it->towerIEta());
        l1upgrade_.tauRawEt.push_back(it->rawEt());
        l1upgrade_.tauIsoEt.push_back(it->isoEt());
        l1upgrade_.tauNTT.push_back(it->nTT());
        l1upgrade_.tauHasEM.push_back(it->hasEM());
        l1upgrade_.tauIsMerged.push_back(it->isMerged());
        l1upgrade_.tauHwQual.push_back(it->hwQual());
        l1upgrade_.nTaus++;
      }
    }
  }
}

void L1Analysis::L1AnalysisL1Upgrade::SetJet(const l1t::JetBxCollection& jet, unsigned maxL1Upgrade) {
  for (int ibx = jet.getFirstBX(); ibx <= jet.getLastBX(); ++ibx) {
    for (l1t::JetBxCollection::const_iterator it = jet.begin(ibx);
         it != jet.end(ibx) && l1upgrade_.nJets < maxL1Upgrade;
         it++) {
      if (it->pt() > 0) {
        l1upgrade_.jetEt.push_back(it->et());
        l1upgrade_.jetEta.push_back(it->eta());
        l1upgrade_.jetPhi.push_back(it->phi());
        l1upgrade_.jetIEt.push_back(it->hwPt());
        l1upgrade_.jetIEta.push_back(it->hwEta());
        l1upgrade_.jetIPhi.push_back(it->hwPhi());
        l1upgrade_.jetHwQual.push_back(it->hwQual());
        l1upgrade_.jetBx.push_back(ibx);
        l1upgrade_.jetRawEt.push_back(it->rawEt());
        l1upgrade_.jetSeedEt.push_back(it->seedEt());
        l1upgrade_.jetTowerIEta.push_back(it->towerIEta());
        l1upgrade_.jetTowerIPhi.push_back(it->towerIPhi());
        l1upgrade_.jetPUEt.push_back(it->puEt());
        l1upgrade_.jetPUDonutEt0.push_back(it->puDonutEt(0));
        l1upgrade_.jetPUDonutEt1.push_back(it->puDonutEt(1));
        l1upgrade_.jetPUDonutEt2.push_back(it->puDonutEt(2));
        l1upgrade_.jetPUDonutEt3.push_back(it->puDonutEt(3));
        l1upgrade_.nJets++;
      }
    }
  }
}

void L1Analysis::L1AnalysisL1Upgrade::SetMuon(const l1t::MuonBxCollection& muon, unsigned maxL1Upgrade) {
  for (int ibx = muon.getFirstBX(); ibx <= muon.getLastBX(); ++ibx) {
    for (l1t::MuonBxCollection::const_iterator it = muon.begin(ibx);
         it != muon.end(ibx) && l1upgrade_.nMuons < maxL1Upgrade;
         it++) {
      if (it->pt() > 0) {
        l1upgrade_.muonEt.push_back(it->et());
        l1upgrade_.muonEtUnconstrained.push_back(it->ptUnconstrained());
        l1upgrade_.muonEta.push_back(it->eta());
        l1upgrade_.muonPhi.push_back(it->phi());
        l1upgrade_.muonEtaAtVtx.push_back(it->etaAtVtx());
        l1upgrade_.muonPhiAtVtx.push_back(it->phiAtVtx());
        l1upgrade_.muonIEt.push_back(it->hwPt());
        l1upgrade_.muonIEtUnconstrained.push_back(it->hwPtUnconstrained());
        l1upgrade_.muonIEta.push_back(it->hwEta());
        l1upgrade_.muonIPhi.push_back(it->hwPhi());
        l1upgrade_.muonIEtaAtVtx.push_back(it->hwEtaAtVtx());
        l1upgrade_.muonIPhiAtVtx.push_back(it->hwPhiAtVtx());
        l1upgrade_.muonIDEta.push_back(it->hwDEtaExtra());
        l1upgrade_.muonIDPhi.push_back(it->hwDPhiExtra());
        l1upgrade_.muonChg.push_back(it->charge());
        l1upgrade_.muonIso.push_back(it->hwIso());
        l1upgrade_.muonQual.push_back(it->hwQual());
        l1upgrade_.muonDxy.push_back(it->hwDXY());
        l1upgrade_.muonTfMuonIdx.push_back(it->tfMuonIndex());
        l1upgrade_.muonBx.push_back(ibx);
        l1upgrade_.nMuons++;
      }
    }
  }
}

void L1Analysis::L1AnalysisL1Upgrade::SetMuonShower(const l1t::MuonShowerBxCollection& muonShower,
                                                    unsigned maxL1Upgrade) {
  for (int ibx = muonShower.getFirstBX(); ibx <= muonShower.getLastBX(); ++ibx) {
    for (l1t::MuonShowerBxCollection::const_iterator it = muonShower.begin(ibx);
         it != muonShower.end(ibx) && l1upgrade_.nMuonShowers < maxL1Upgrade;
         it++) {
      if (it->isValid()) {
        l1upgrade_.muonShowerBx.push_back(ibx);
        l1upgrade_.muonShowerOneNominal.push_back(it->isOneNominalInTime());
        l1upgrade_.muonShowerOneTight.push_back(it->isOneTightInTime());
        l1upgrade_.muonShowerTwoLoose.push_back(it->isTwoLooseInTime());
        l1upgrade_.muonShowerTwoLooseDiffSectors.push_back(it->isTwoLooseDiffSectorsInTime());
        l1upgrade_.nMuonShowers++;
      }
    }
  }
}

void L1Analysis::L1AnalysisL1Upgrade::SetSum(const l1t::EtSumBxCollection& sums, unsigned maxL1Upgrade) {
  for (int ibx = sums.getFirstBX(); ibx <= sums.getLastBX(); ++ibx) {
    for (l1t::EtSumBxCollection::const_iterator it = sums.begin(ibx);
         it != sums.end(ibx) && l1upgrade_.nSums < maxL1Upgrade;
         it++) {
      int type = static_cast<int>(it->getType());
      l1upgrade_.sumType.push_back(type);
      l1upgrade_.sumEt.push_back(it->et());
      l1upgrade_.sumPhi.push_back(it->phi());
      l1upgrade_.sumIEt.push_back(it->hwPt());
      l1upgrade_.sumIPhi.push_back(it->hwPhi());
      l1upgrade_.sumBx.push_back(ibx);
      l1upgrade_.nSums++;
    }
  }
}
