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
      egIso.clear();
      egBx.clear();

      nTaus = 0;
      tauEt.clear();
      tauEta.clear();
      tauPhi.clear(); 
      tauIso.clear();
      tauBx.clear();

      nJets = 0;
      jetEt.clear();
      jetEta.clear();
      jetPhi.clear();
      jetBx.clear();

      nMuons = 0;
      muonEt.clear();
      muonEta.clear();
      muonPhi.clear();
      muonChg.clear();
      muonIso.clear();
      muonBx.clear();

      nSums = 0;
      sumEt.clear();
      sumPhi.clear();
      sumBx.clear();

    }
   
    unsigned int nEGs;
    std::vector<double> egEt;
    std::vector<double> egEta;
    std::vector<double> egPhi;
    std::vector<int>    egIso;
    std::vector<int>    egBx;
 
    unsigned int nTaus;
    std::vector<double> tauEt;
    std::vector<double> tauEta;
    std::vector<double> tauPhi;
    std::vector<int>    tauIso;
    std::vector<int>    tauBx;

    unsigned int nJets;
    std::vector<double> jetEt;
    std::vector<double> jetEta;
    std::vector<double> jetPhi;
    std::vector<int>    jetBx;

    unsigned int nMuons;
    std::vector<double>   muonEt;
    std::vector<double>   muonEta;
    std::vector<double>   muonPhi;
    std::vector<int>      muonChg;
    std::vector<unsigned int> muonIso;
    std::vector<int>      muonBx;

    
    unsigned int nSums;
    std::vector<double> sumEt;
    std::vector<double> sumPhi;
    std::vector<double> sumBx;

  }; 
}
#endif


