#ifndef __L1Analysis_L1AnalysisL1UpgradeDataFormat_H__
#define __L1Analysis_L1AnalysisL1UpgradeDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1TriggerDPG/L1Ntuples/L1UpgradeTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------


#include <vector>

namespace L1Analysis
{

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
  };

  struct L1AnalysisL1UpgradeDataFormat
  {
  
    L1AnalysisL1UpgradeDataFormat(){ Reset();};
    ~L1AnalysisL1UpgradeDataFormat(){};
    
    void Reset()
    {
      nEGs = 0;
      egEt.clear();
      egEta.clear();
      egPhi.clear();
      egIEt.clear();
      egIEta.clear();
      egIPhi.clear();
      egIso.clear();
      egBx.clear();

      nTaus = 0;
      tauEt.clear();
      tauEta.clear();
      tauPhi.clear(); 
      tauIEt.clear();
      tauIEta.clear();
      tauIPhi.clear(); 
      tauIso.clear();
      tauBx.clear();

      nJets = 0;
      jetEt.clear();
      jetEta.clear();
      jetPhi.clear();
      jetIEt.clear();
      jetIEta.clear();
      jetIPhi.clear();
      jetBx.clear();

      nMuons = 0;
      muonEt.clear();
      muonEta.clear();
      muonPhi.clear();
      muonIEt.clear();
      muonIEta.clear();
      muonIPhi.clear();
      muonChg.clear();
      muonIso.clear();
      muonQual.clear();
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
    std::vector<short int>    egIso;
    std::vector<short int>    egBx;
 
    unsigned short int nTaus;
    std::vector<float> tauEt;
    std::vector<float> tauEta;
    std::vector<float> tauPhi;
    std::vector<short int> tauIEt;
    std::vector<short int> tauIEta;
    std::vector<short int> tauIPhi;
    std::vector<short int>    tauIso;
    std::vector<short int>    tauBx;

    unsigned short int nJets;
    std::vector<float> jetEt;
    std::vector<float> jetEta;
    std::vector<float> jetPhi;
    std::vector<short int> jetIEt;
    std::vector<short int> jetIEta;
    std::vector<short int> jetIPhi;
    std::vector<short int>    jetBx;

    unsigned short int nMuons;
    std::vector<float>   muonEt;
    std::vector<float>   muonEta;
    std::vector<float>   muonPhi;
    std::vector<short int>   muonIEt;
    std::vector<short int>   muonIEta;
    std::vector<short int>   muonIPhi;
    std::vector<short int>      muonChg;
    std::vector<unsigned short int> muonIso;
    std::vector<unsigned short int> muonQual;
    std::vector<unsigned short int> muonTfMuonIdx;
    std::vector<short int>      muonBx;

    
    unsigned short int nSums;
    std::vector<short int> sumType;
    std::vector<float> sumEt;
    std::vector<float> sumPhi;
    std::vector<short int> sumIEt;
    std::vector<short int> sumIPhi;
    std::vector<float> sumBx;

  }; 
}
#endif


