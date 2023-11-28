#ifndef __L1Analysis_L1AnalysisRecoJetDataFormat_H__
#define __L1Analysis_L1AnalysisRecoJetDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
//
//
// Original code : L1Trigger/L1TNtuples/L1RecoJetNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis {
  struct L1AnalysisRecoJetDataFormat {
    L1AnalysisRecoJetDataFormat() { Reset(); };
    ~L1AnalysisRecoJetDataFormat() { Reset(); };

    void Reset() {
      nJets = 0;

      e.clear();
      et.clear();
      etCorr.clear();
      corrFactor.clear();
      eta.clear();
      phi.clear();

      puppi_nJets = 0;
      puppi_nUncorrJets = 0;

      puppi_e.clear();
      puppi_et.clear();
      puppi_etCorr.clear();
      puppi_eta.clear();
      puppi_phi.clear();

      nCaloJets = 0;

      caloE.clear();
      caloEt.clear();
      caloEtCorr.clear();
      caloCorrFactor.clear();
      caloEta.clear();
      caloPhi.clear();

      // calo quantities
      eEMF.clear();
      eEmEB.clear();
      eEmEE.clear();
      eEmHF.clear();
      eHadHB.clear();
      eHadHE.clear();
      eHadHO.clear();
      eHadHF.clear();
      eMaxEcalTow.clear();
      eMaxHcalTow.clear();
      towerArea.clear();
      towerSize.clear();
      n60.clear();
      n90.clear();
      n90hits.clear();
      fHPD.clear();
      fRBX.clear();

      // PF quantities
      chef.clear();
      nhef.clear();
      pef.clear();
      eef.clear();
      mef.clear();
      hfhef.clear();
      hfemef.clear();
      chMult.clear();
      nhMult.clear();
      phMult.clear();
      elMult.clear();
      muMult.clear();
      hfhMult.clear();
      hfemMult.clear();

      cemef.clear();
      cmef.clear();
      nemef.clear();
      cMult.clear();
      nMult.clear();

      // PUPPI quantities
      puppi_chef.clear();
      puppi_nhef.clear();
      puppi_pef.clear();
      puppi_eef.clear();
      puppi_mef.clear();
      puppi_hfhef.clear();
      puppi_hfemef.clear();
      puppi_chMult.clear();
      puppi_nhMult.clear();
      puppi_phMult.clear();
      puppi_elMult.clear();
      puppi_muMult.clear();
      puppi_hfhMult.clear();
      puppi_hfemMult.clear();

      puppi_cemef.clear();
      puppi_cmef.clear();
      puppi_nemef.clear();
      puppi_cMult.clear();
      puppi_nMult.clear();
    }

    unsigned short nJets;
    std::vector<float> e;
    std::vector<float> et;
    std::vector<float> etCorr;
    std::vector<float> corrFactor;
    std::vector<float> eta;
    std::vector<float> phi;

    unsigned short puppi_nJets;
    unsigned short puppi_nUncorrJets;
    std::vector<float> puppi_e;
    std::vector<float> puppi_et;
    std::vector<float> puppi_etCorr;
    std::vector<float> puppi_eta;
    std::vector<float> puppi_phi;

    unsigned short nCaloJets;
    std::vector<float> caloE;
    std::vector<float> caloEt;
    std::vector<float> caloEtCorr;
    std::vector<float> caloCorrFactor;
    std::vector<float> caloEta;
    std::vector<float> caloPhi;

    std::vector<float> eEMF;
    std::vector<float> eHadHB;
    std::vector<float> eHadHE;
    std::vector<float> eHadHO;
    std::vector<float> eHadHF;
    std::vector<float> eEmEB;
    std::vector<float> eEmEE;
    std::vector<float> eEmHF;
    std::vector<float> eMaxEcalTow;
    std::vector<float> eMaxHcalTow;
    std::vector<float> towerArea;
    std::vector<short> towerSize;
    std::vector<short> n60;
    std::vector<short> n90;

    std::vector<short> n90hits;
    std::vector<float> fHPD;
    std::vector<float> fRBX;

    std::vector<float> chef;
    std::vector<float> nhef;
    std::vector<float> pef;
    std::vector<float> eef;
    std::vector<float> mef;
    std::vector<float> hfhef;
    std::vector<float> hfemef;
    std::vector<short> chMult;
    std::vector<short> nhMult;
    std::vector<short> phMult;
    std::vector<short> elMult;
    std::vector<short> muMult;
    std::vector<short> hfhMult;
    std::vector<short> hfemMult;

    std::vector<float> cemef;
    std::vector<float> cmef;
    std::vector<float> nemef;
    std::vector<int> cMult;
    std::vector<int> nMult;

    std::vector<float> puppi_chef;
    std::vector<float> puppi_nhef;
    std::vector<float> puppi_pef;
    std::vector<float> puppi_eef;
    std::vector<float> puppi_mef;
    std::vector<float> puppi_hfhef;
    std::vector<float> puppi_hfemef;
    std::vector<short> puppi_chMult;
    std::vector<short> puppi_nhMult;
    std::vector<short> puppi_phMult;
    std::vector<short> puppi_elMult;
    std::vector<short> puppi_muMult;
    std::vector<short> puppi_hfhMult;
    std::vector<short> puppi_hfemMult;

    std::vector<float> puppi_cemef;
    std::vector<float> puppi_cmef;
    std::vector<float> puppi_nemef;
    std::vector<int> puppi_cMult;
    std::vector<int> puppi_nMult;
  };
}  // namespace L1Analysis
#endif
