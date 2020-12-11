#ifndef __L1Analysis_L1AnalysisL1UpgradeDataFormat_H__
#define __L1Analysis_L1AnalysisL1UpgradeDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1TriggerDPG/L1Ntuples/L1UpgradeTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis {

  // copied from DataFormats/L1Trigger/interface/EtSum.h, for use in standalone ROOT macros which use this class.
  enum EtSumType {
    kTotalEt,
    kTotalHt,
    kMissingEt,
    kMissingHt,
    kTotalEtx,
    kTotalEty,
    kTotalHtx,
    kTotalHty,
    kMissingEtHF,
    kTotalEtxHF,
    kTotalEtyHF,
    kMinBiasHFP0,
    kMinBiasHFM0,
    kMinBiasHFP1,
    kMinBiasHFM1,
    kTotalEtHF,
    kTotalEtEm,
    kTotalHtHF,
    kTotalHtxHF,
    kTotalHtyHF,
    kMissingHtHF,
    kTowerCount,
    kCentrality,
    kAsymEt,
    kAsymHt,
    kAsymEtHF,
    kAsymHtHF
  };

  struct L1AnalysisL1UpgradeDataFormat {
    L1AnalysisL1UpgradeDataFormat() { Reset(); };
    ~L1AnalysisL1UpgradeDataFormat(){};

    void Reset() {
      nEGs = 0;
      egEt.clear();
      egEta.clear();
      egPhi.clear();
      egIEt.clear();
      egIEta.clear();
      egIPhi.clear();
      egIso.clear();
      egBx.clear();
      egTowerIPhi.clear();
      egTowerIEta.clear();
      egRawEt.clear();
      egIsoEt.clear();
      egFootprintEt.clear();
      egNTT.clear();
      egShape.clear();
      egTowerHoE.clear();
      egHwQual.clear();

      nTaus = 0;
      tauEt.clear();
      tauEta.clear();
      tauPhi.clear();
      tauIEt.clear();
      tauIEta.clear();
      tauIPhi.clear();
      tauIso.clear();
      tauBx.clear();
      tauTowerIPhi.clear();
      tauTowerIEta.clear();
      tauRawEt.clear();
      tauIsoEt.clear();
      tauNTT.clear();
      tauHasEM.clear();
      tauIsMerged.clear();
      tauHwQual.clear();

      nJets = 0;
      jetEt.clear();
      jetEta.clear();
      jetPhi.clear();
      jetIEt.clear();
      jetIEta.clear();
      jetIPhi.clear();
      jetBx.clear();
      jetTowerIPhi.clear();
      jetTowerIEta.clear();
      jetRawEt.clear();
      jetSeedEt.clear();
      jetPUEt.clear();
      jetPUDonutEt0.clear();
      jetPUDonutEt1.clear();
      jetPUDonutEt2.clear();
      jetPUDonutEt3.clear();

      nMuons = 0;
      muonEt.clear();
      muonEtUnconstrained.clear();
      muonEta.clear();
      muonPhi.clear();
      muonEtaAtVtx.clear();
      muonPhiAtVtx.clear();
      muonIEt.clear();
      muonIEtUnconstrained.clear();
      muonIEta.clear();
      muonIPhi.clear();
      muonIEtaAtVtx.clear();
      muonIPhiAtVtx.clear();
      muonIDEta.clear();
      muonIDPhi.clear();
      muonChg.clear();
      muonIso.clear();
      muonQual.clear();
      muonDxy.clear();
      muonTfMuonIdx.clear();
      muonBx.clear();

      nSums = 0;
      sumType.clear();
      sumEt.clear();
      sumPhi.clear();
      sumIEt.clear();
      sumIPhi.clear();
      sumBx.clear();
    }

    unsigned short int nEGs;
    std::vector<float> egEt;
    std::vector<float> egEta;
    std::vector<float> egPhi;
    std::vector<short int> egIEt;
    std::vector<short int> egIEta;
    std::vector<short int> egIPhi;
    std::vector<short int> egIso;
    std::vector<short int> egBx;
    std::vector<short int> egTowerIPhi;
    std::vector<short int> egTowerIEta;
    std::vector<short int> egRawEt;
    std::vector<short int> egIsoEt;
    std::vector<short int> egFootprintEt;
    std::vector<short int> egNTT;
    std::vector<short int> egShape;
    std::vector<short int> egTowerHoE;
    std::vector<short int> egHwQual;

    unsigned short int nTaus;
    std::vector<float> tauEt;
    std::vector<float> tauEta;
    std::vector<float> tauPhi;
    std::vector<short int> tauIEt;
    std::vector<short int> tauIEta;
    std::vector<short int> tauIPhi;
    std::vector<short int> tauIso;
    std::vector<short int> tauBx;
    std::vector<short int> tauTowerIPhi;
    std::vector<short int> tauTowerIEta;
    std::vector<short int> tauRawEt;
    std::vector<short int> tauIsoEt;
    std::vector<short int> tauNTT;
    std::vector<short int> tauHasEM;
    std::vector<short int> tauIsMerged;
    std::vector<short int> tauHwQual;

    unsigned short int nJets;
    std::vector<float> jetEt;
    std::vector<float> jetEta;
    std::vector<float> jetPhi;
    std::vector<short int> jetIEt;
    std::vector<short int> jetIEta;
    std::vector<short int> jetIPhi;
    std::vector<short int> jetBx;
    std::vector<short int> jetTowerIPhi;
    std::vector<short int> jetTowerIEta;
    std::vector<short int> jetRawEt;
    std::vector<short int> jetSeedEt;
    std::vector<short int> jetPUEt;
    std::vector<short int> jetPUDonutEt0;
    std::vector<short int> jetPUDonutEt1;
    std::vector<short int> jetPUDonutEt2;
    std::vector<short int> jetPUDonutEt3;

    unsigned short int nMuons;
    std::vector<float> muonEt;
    std::vector<float> muonEtUnconstrained;
    std::vector<float> muonEta;
    std::vector<float> muonPhi;
    std::vector<float> muonEtaAtVtx;
    std::vector<float> muonPhiAtVtx;
    std::vector<short int> muonIEt;
    std::vector<short int> muonIEtUnconstrained;
    std::vector<short int> muonIEta;
    std::vector<short int> muonIPhi;
    std::vector<short int> muonIEtaAtVtx;
    std::vector<short int> muonIPhiAtVtx;
    std::vector<short int> muonIDEta;
    std::vector<short int> muonIDPhi;
    std::vector<short int> muonChg;
    std::vector<unsigned short int> muonIso;
    std::vector<unsigned short int> muonQual;
    std::vector<unsigned short int> muonDxy;
    std::vector<unsigned short int> muonTfMuonIdx;
    std::vector<short int> muonBx;

    unsigned short int nSums;
    std::vector<short int> sumType;
    std::vector<float> sumEt;
    std::vector<float> sumPhi;
    std::vector<short int> sumIEt;
    std::vector<short int> sumIPhi;
    std::vector<float> sumBx;
  };
}  // namespace L1Analysis
#endif
