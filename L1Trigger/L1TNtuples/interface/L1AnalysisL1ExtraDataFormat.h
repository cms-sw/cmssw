#ifndef __L1Analysis_L1AnalysisL1ExtraDataFormat_H__
#define __L1Analysis_L1AnalysisL1ExtraDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1ExtraTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------


#include <vector>

namespace L1Analysis
{
  struct L1AnalysisL1ExtraDataFormat
  {
    L1AnalysisL1ExtraDataFormat(){Reset();};
    ~L1AnalysisL1ExtraDataFormat(){};
    
    void Reset()
    {
      nIsoEm = 0;
      isoEmEt.clear();
      isoEmEta.clear();
      isoEmPhi.clear();
      isoEmBx.clear();

      nNonIsoEm = 0;
      nonIsoEmEt.clear();
      nonIsoEmEta.clear();
      nonIsoEmPhi.clear();
      nonIsoEmBx.clear();
      
      nCenJets = 0;
      cenJetEt.clear();
      cenJetEta.clear();
      cenJetPhi.clear();
      cenJetBx.clear();

      nFwdJets = 0;
      fwdJetEt.clear();
      fwdJetEta.clear();
      fwdJetPhi.clear();
      fwdJetBx.clear();

      nTauJets = 0;
      tauJetEt.clear();
      tauJetEta.clear();
      tauJetPhi.clear(); 
      tauJetBx.clear();

      nIsoTauJets = 0;
      isoTauJetEt.clear();
      isoTauJetEta.clear();
      isoTauJetPhi.clear(); 
      isoTauJetBx.clear();

      nMuons = 0;
      muonEt.clear();
      muonEta.clear();
      muonPhi.clear();
      muonChg.clear();
      muonIso.clear();
      muonFwd.clear();
      muonMip.clear();
      muonRPC.clear();
      muonBx.clear();
      muonQuality.clear();

      nMet = 0;
      et.clear();
      met.clear();
      metPhi.clear();
      metBx.clear();

      nMht = 0;
      ht.clear();
      mht.clear();
      mhtPhi.clear();
      mhtBx.clear();

      hfEtSum.clear();
      hfBitCnt.clear();
      hfBx.clear();

    }
   
    unsigned short nIsoEm;
    std::vector<float> isoEmEt;
    std::vector<float> isoEmEta;
    std::vector<float> isoEmPhi;
    std::vector<int>    isoEmBx;
 
    unsigned short nNonIsoEm;
    std::vector<float> nonIsoEmEt;
    std::vector<float> nonIsoEmEta;
    std::vector<float> nonIsoEmPhi;
    std::vector<int>    nonIsoEmBx;
 
    unsigned short nCenJets;
    std::vector<float> cenJetEt;
    std::vector<float> cenJetEta;
    std::vector<float> cenJetPhi;
    std::vector<int>    cenJetBx;
 
    unsigned short nFwdJets;
    std::vector<float> fwdJetEt;
    std::vector<float> fwdJetEta;
    std::vector<float> fwdJetPhi;
    std::vector<int>    fwdJetBx;

    unsigned short nTauJets;
    std::vector<float> tauJetEt;
    std::vector<float> tauJetEta;
    std::vector<float> tauJetPhi;
    std::vector<int>    tauJetBx;

    unsigned short nIsoTauJets;
    std::vector<float> isoTauJetEt;
    std::vector<float> isoTauJetEta;
    std::vector<float> isoTauJetPhi;
    std::vector<int>    isoTauJetBx;

    unsigned short nMuons;
    std::vector<float>   muonEt;
    std::vector<float>   muonEta;
    std::vector<float>   muonPhi;
    std::vector<int>      muonChg;
    std::vector<unsigned short> muonIso;
    std::vector<unsigned short> muonFwd;
    std::vector<unsigned short> muonMip;
    std::vector<unsigned short> muonRPC;
    std::vector<int>      muonBx;
    std::vector<int>      muonQuality;
 
    std::vector<float> hfEtSum;
    std::vector<unsigned short> hfBitCnt;
    std::vector<int> hfBx;
    
    unsigned short nMet;
    std::vector<float> et;
    std::vector<float> met;
    std::vector<float> metPhi;
    std::vector<float> metBx;

    unsigned short nMht;
    std::vector<float> ht;
    std::vector<float> mht;
    std::vector<float> mhtPhi;
    std::vector<float> mhtBx;

  }; 
}
#endif


