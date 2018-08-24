#ifndef __L1Analysis_L1AnalysisPhaseIIDataFormat_H__
#define __L1Analysis_L1AnalysisPhaseIIDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
// 
// Original code : UserCode/L1TriggerDPG/L1ExtraTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------


#include <vector>

namespace L1Analysis
{
  struct L1AnalysisPhaseIIDataFormat
  {
    L1AnalysisPhaseIIDataFormat(){Reset();};
    ~L1AnalysisPhaseIIDataFormat(){};
    
    void Reset()
    {

      z0Puppi=0;
      z0VertexTDR=0;
      z0Vertices.clear();
      z0L1TkPV.clear();
      sumL1TkPV.clear();
      nL1TkPVs=0;
      nVertices=0;

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
      muonEta.clear();
      muonPhi.clear();
      muonEtaAtVtx.clear();
      muonPhiAtVtx.clear();
      muonIEt.clear();
      muonIEta.clear();
      muonIPhi.clear();
      muonIEtaAtVtx.clear();
      muonIPhiAtVtx.clear();
      muonIDEta.clear();
      muonIDPhi.clear();
      muonChg.clear();
      muonIso.clear();
      muonQual.clear();
      muonTfMuonIdx.clear();
      muonBx.clear();


      nMuonsKF = 0;
      muonKFEt.clear();
      muonKFEta.clear();
      muonKFPhi.clear();
      muonKFChg.clear();
      muonKFQual.clear();
      muonKFBx.clear();



      nSums = 0;
      sumType.clear();
      sumEt.clear();
      sumPhi.clear();
      sumIEt.clear();
      sumIPhi.clear();
      sumBx.clear(); 

      nEG = 0;
      EGEt.clear();
      EGEta.clear();
      EGPhi.clear();
      EGBx.clear();
      EGIso.clear();
      EGzVtx.clear();
      EGHwQual.clear();      
 
      nTkEG = 0;
      tkEGEt.clear();
      tkEGEta.clear();
      tkEGPhi.clear();
      tkEGBx.clear();
      tkEGTrkIso.clear();
      tkEGzVtx.clear();
      tkEGHwQual.clear();
      tkEGEGRefPt.clear(); 
      tkEGEGRefEta.clear();
      tkEGEGRefPhi.clear();



      ntkEGLoose = 0;
      tkEGLooseEt.clear();
      tkEGLooseEta.clear();
      tkEGLoosePhi.clear();
      tkEGLooseBx.clear();
      tkEGLooseTrkIso.clear();
      tkEGLoosezVtx.clear();
      tkEGLooseHwQual.clear();
      tkEGLooseEGRefPt.clear();
      tkEGLooseEGRefEta.clear();
      tkEGLooseEGRefPhi.clear();

      nTkEM = 0;
      tkEMEt.clear();
      tkEMEta.clear();
      tkEMPhi.clear();
      tkEMBx.clear();
      tkEMTrkIso.clear();
      tkEMzVtx.clear();
      tkEMHwQual.clear();
      tkEMEGRefPt.clear();
      tkEMEGRefEta.clear();
      tkEMEGRefPhi.clear();

      // TkTaus
      nTkTau = 0;
      tkTauEt.clear();
      tkTauEta.clear();
      tkTauPhi.clear();
      tkTauBx.clear();
      tkTauTrkIso.clear();
      tkTauzVtx.clear();

      // TkJets
      nTrackerJets = 0;
      trackerJetEt.clear();
      trackerJetEta.clear();
      trackerJetPhi.clear();
      trackerJetBx.clear();
      trackerJetzVtx.clear();

      // TkCaloJets
      nTkCaloJets = 0;
      tkCaloJetEt.clear();
      tkCaloJetEta.clear();
      tkCaloJetPhi.clear();
      tkCaloJetBx.clear();

      // tkTkGlbMuons
      nTkGlbMuons = 0;
      tkGlbMuonEt.clear();
      tkGlbMuonEta.clear();
      tkGlbMuonPhi.clear();
      tkGlbMuonChg.clear();
      tkGlbMuonTrkIso.clear();
      tkGlbMuonBx.clear();
      tkGlbMuonQuality.clear();
      tkGlbMuonzVtx.clear();
      tkGlbMuonMuRefPt.clear();
      tkGlbMuonTrkRefPt.clear();
      tkGlbMuonMuRefPhi.clear();
      tkGlbMuonMuRefEta.clear();
      tkGlbMuonDRMuTrack.clear();
      tkGlbMuonNMatchedTracks.clear();

      nTkMuons = 0;
      tkMuonEt.clear();
      tkMuonEta.clear();
      tkMuonPhi.clear();
      tkMuonChg.clear();
      tkMuonTrkIso.clear();
      tkMuonBx.clear();
      tkMuonQuality.clear();
      tkMuonzVtx.clear();
      tkMuonMuRefPt.clear();
      tkMuonTrkRefPt.clear();
      tkMuonMuRefPhi.clear(); 
      tkMuonMuRefEta.clear();
      tkMuonDRMuTrack.clear();
      tkMuonNMatchedTracks.clear();

      // TrackerMet
      nTrackerMet = 0;
      trackerMetSumEt.clear();
      trackerMetEt.clear();
      trackerMetPhi.clear();
      trackerMetBx.clear();

      //trackerMHT
      nTrackerMHT = 0;
      trackerHT.clear();
      trackerMHT.clear();
      trackerMHTPhi.clear();

      // New Jet Collections
      nPuppiJets = 0;
      puppiJetEt.clear();
      puppiJetEta.clear();
      puppiJetPhi.clear();
      puppiJetBx.clear();
      puppiJetzVtx.clear();
      puppiJetEtUnCorr.clear();

      puppiMETEt=0;
      puppiMETPhi=0;
      puppiHT.clear();
      puppiMHTEt.clear();
      puppiMHTPhi.clear();
      nPuppiMHT=0;
    }
 
    float z0Puppi;
    float z0VertexTDR;
    unsigned short int nVertices;
    std::vector<float>z0Vertices;  
    unsigned short int nL1TkPVs;
    std::vector<float>  z0L1TkPV;
    std::vector<float>  sumL1TkPV;
 
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
    std::vector<float>   muonEt;
    std::vector<float>   muonEta;
    std::vector<float>   muonPhi;
    std::vector<float>   muonEtaAtVtx;
    std::vector<float>   muonPhiAtVtx;
    std::vector<short int>   muonIEt;
    std::vector<short int>   muonIEta;
    std::vector<short int>   muonIPhi;
    std::vector<short int>   muonIEtaAtVtx;
    std::vector<short int>   muonIPhiAtVtx;
    std::vector<short int>   muonIDEta;
    std::vector<short int>   muonIDPhi;
    std::vector<short int>      muonChg;
    std::vector<unsigned short int> muonIso;
    std::vector<unsigned short int> muonQual;
    std::vector<unsigned short int> muonTfMuonIdx;
    std::vector<short int>      muonBx;


    unsigned short int nMuonsKF;
    std::vector<float>   muonKFEt;
    std::vector<float>   muonKFEta;
    std::vector<float>   muonKFPhi;
    std::vector<short int>      muonKFChg;
    std::vector<unsigned short int> muonKFQual;
    std::vector<short int>      muonKFBx;

    unsigned short int nSums;
    std::vector<short int> sumType;
    std::vector<float> sumEt;
    std::vector<float> sumPhi;
    std::vector<short int> sumIEt;
    std::vector<short int> sumIPhi;
    std::vector<float> sumBx;

 
    unsigned int nEG;
    std::vector<double> EGEt;
    std::vector<double> EGEta;
    std::vector<double> EGPhi;
    std::vector<int>    EGBx;
    std::vector<double> EGIso;
    std::vector<double> EGzVtx;
    std::vector<int>    EGHwQual;

    unsigned int nTkEG;
    std::vector<double> tkEGEt;
    std::vector<double> tkEGEta;
    std::vector<double> tkEGPhi;
    std::vector<int>    tkEGBx;
    std::vector<double> tkEGTrkIso;
    std::vector<double> tkEGzVtx;
    std::vector<double> tkEGHwQual;
    std::vector<double>   tkEGEGRefPt;
    std::vector<double>   tkEGEGRefEta;
    std::vector<double>   tkEGEGRefPhi;


    unsigned int ntkEGLoose;
    std::vector<double> tkEGLooseEt;
    std::vector<double> tkEGLooseEta;
    std::vector<double> tkEGLoosePhi;
    std::vector<int>    tkEGLooseBx;
    std::vector<double> tkEGLooseTrkIso;
    std::vector<double> tkEGLoosezVtx;
    std::vector<double> tkEGLooseHwQual;
    std::vector<double>   tkEGLooseEGRefPt;
    std::vector<double>   tkEGLooseEGRefEta;
    std::vector<double>   tkEGLooseEGRefPhi;

    unsigned int nTkEM;
    std::vector<double> tkEMEt;
    std::vector<double> tkEMEta;
    std::vector<double> tkEMPhi;
    std::vector<int>    tkEMBx;
    std::vector<double> tkEMTrkIso;
    std::vector<double> tkEMzVtx;
    std::vector<double> tkEMHwQual;
    std::vector<double>   tkEMEGRefPt;
    std::vector<double>   tkEMEGRefEta;
    std::vector<double>   tkEMEGRefPhi;

    unsigned int nTkTau;
    std::vector<double> tkTauEt;
    std::vector<double> tkTauEta;
    std::vector<double> tkTauPhi;
    std::vector<int>    tkTauBx;
    std::vector<double> tkTauTrkIso;
    std::vector<double> tkTauzVtx;

    unsigned int nTrackerJets;
    std::vector<double> trackerJetEt;
    std::vector<double> trackerJetEta;
    std::vector<double> trackerJetPhi;
    std::vector<int>    trackerJetBx;
    std::vector<double> trackerJetzVtx;

    unsigned int nTkCaloJets;
    std::vector<double> tkCaloJetEt;
    std::vector<double> tkCaloJetEta;
    std::vector<double> tkCaloJetPhi;
    std::vector<int>    tkCaloJetBx;
    std::vector<double> tkCaloJetzVtx;

    unsigned int nTkGlbMuons;
    std::vector<double>   tkGlbMuonEt;
    std::vector<double>   tkGlbMuonEta;
    std::vector<double>   tkGlbMuonPhi;
    std::vector<int>      tkGlbMuonChg;
    std::vector<unsigned int> tkGlbMuonIso;
    std::vector<double> tkGlbMuonTrkIso;
    std::vector<int>      tkGlbMuonBx;
    std::vector<unsigned int>      tkGlbMuonQuality;
    std::vector<double>   tkGlbMuonzVtx;
    std::vector<double> tkGlbMuonMuRefPt;
    std::vector<double> tkGlbMuonTrkRefPt;
    std::vector<double>  tkGlbMuonMuRefPhi;
    std::vector<double>  tkGlbMuonMuRefEta;
    std::vector<double>  tkGlbMuonDRMuTrack;
    std::vector<double>  tkGlbMuonNMatchedTracks;

    unsigned int nTkMuons;
    std::vector<double>   tkMuonEt;
    std::vector<double>   tkMuonEta;
    std::vector<double>   tkMuonPhi;
    std::vector<int>      tkMuonChg;
    std::vector<unsigned int> tkMuonIso;
    std::vector<double> tkMuonTrkIso;
    std::vector<unsigned int> tkMuonFwd;
    std::vector<unsigned int> tkMuonMip;
    std::vector<unsigned int> tkMuonRPC;
    std::vector<int>      tkMuonBx;
    std::vector<unsigned int>      tkMuonQuality;
    std::vector<double>   tkMuonzVtx;
    std::vector<double> tkMuonMuRefPt;
    std::vector<double> tkMuonTrkRefPt;
    std::vector<double>  tkMuonMuRefPhi;
    std::vector<double>  tkMuonMuRefEta;
    std::vector<double>  tkMuonDRMuTrack;
    std::vector<double>  tkMuonNMatchedTracks;

    unsigned int nTrackerMet;
    std::vector<double> trackerMetSumEt;
    std::vector<double> trackerMetEt;
    std::vector<double> trackerMetPhi;
    std::vector<double> trackerMetBx;

    unsigned int nTrackerMHT;
    std::vector<double> trackerHT;
    std::vector<double> trackerMHT;
    std::vector<double> trackerMHTPhi;


    unsigned int nPuppiJets;
    std::vector<double> puppiJetEt;
    std::vector<double> puppiJetEta;
    std::vector<double> puppiJetPhi;
    std::vector<int>    puppiJetBx;
    std::vector<double> puppiJetzVtx;
    std::vector<double> puppiJetEtUnCorr;

    double puppiMETEt;
    double puppiMETPhi;

    std::vector<double> puppiHT;
    std::vector<double> puppiMHTEt;
    std::vector<double> puppiMHTPhi;
     unsigned int nPuppiMHT;

  }; 
}
#endif


