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

      nGlobalMuons = 0;
      globalMuonPt.clear();
      globalMuonEta.clear();
      globalMuonPhi.clear();
      globalMuonEtaAtVtx.clear();
      globalMuonPhiAtVtx.clear();
      globalMuonIEt.clear();
      globalMuonIEta.clear();
      globalMuonIPhi.clear();
      globalMuonIEtaAtVtx.clear();
      globalMuonIPhiAtVtx.clear();
      globalMuonIDEta.clear();
      globalMuonIDPhi.clear();
      globalMuonChg.clear();
      globalMuonIso.clear();
      globalMuonQual.clear();
      globalMuonTfMuonIdx.clear();
      globalMuonBx.clear();


      nStandaloneMuons = 0;
      standaloneMuonPt.clear();
      standaloneMuonPt2.clear();
      standaloneMuonEta.clear();
      standaloneMuonPhi.clear();
      standaloneMuonChg.clear();
      standaloneMuonQual.clear();
      standaloneMuonBx.clear();
      standaloneMuonRegion.clear();



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
      EGHGC.clear();
      EGPassesID.clear();

      nTkElectrons = 0;
      tkElectronEt.clear();
      tkElectronEta.clear();
      tkElectronPhi.clear();
      tkElectronChg.clear();
      tkElectronBx.clear();
      tkElectronTrkIso.clear();
      tkElectronzVtx.clear();
      tkElectronHwQual.clear();
      tkElectronEGRefPt.clear(); 
      tkElectronEGRefEta.clear();
      tkElectronEGRefPhi.clear();
      tkElectronHGC.clear();
      tkElectronPassesID.clear();



      nTkElectronsLoose = 0;
      tkElectronLooseEt.clear();
      tkElectronLooseEta.clear();
      tkElectronLoosePhi.clear();
      tkElectronLooseChg.clear();
      tkElectronLooseBx.clear();
      tkElectronLooseTrkIso.clear();
      tkElectronLoosezVtx.clear();
      tkElectronLooseHwQual.clear();
      tkElectronLooseEGRefPt.clear();
      tkElectronLooseEGRefEta.clear();
      tkElectronLooseEGRefPhi.clear();
      tkElectronLooseHGC.clear();
      tkElectronLoosePassesID.clear();         

      nTkPhotons = 0;
      tkPhotonEt.clear();
      tkPhotonEta.clear();
      tkPhotonPhi.clear();
      tkPhotonBx.clear();
      tkPhotonTrkIso.clear();
      tkPhotonzVtx.clear();
      tkPhotonHwQual.clear();
      tkPhotonEGRefPt.clear();
      tkPhotonEGRefEta.clear();
      tkPhotonEGRefPhi.clear();
      tkPhotonHGC.clear();
      tkPhotonPassesID.clear();

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
      tkGlbMuonPt.clear();
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
      tkMuonPt.clear();
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
      tkMuonMuRefChg.clear();
      tkMuonRegion.clear();

      nTkMuonStubs = 0;
      tkMuonStubsPt.clear();
      tkMuonStubsEta.clear();
      tkMuonStubsPhi.clear();
      tkMuonStubsChg.clear();
      tkMuonStubsTrkIso.clear();
      tkMuonStubsBx.clear();
      tkMuonStubsQuality.clear();
      tkMuonStubszVtx.clear();
      tkMuonStubsRegion.clear();
      tkMuonStubsBarrelStubs.clear();
      tkMuonStubsRegion.clear();

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

      nPFMuons = 0;
      pfMuonPt.clear();
      pfMuonEta.clear();
      pfMuonPhi.clear();
      pfMuonzVtx.clear();

      nPFCands = 0;
      pfCandId.clear();
      pfCandEt.clear();
      pfCandEta.clear();
      pfCandPhi.clear();
      pfCandzVtx.clear();

      nPFTaus = 0; 
      pfTauEt.clear();
      pfTauEta.clear();
      pfTauPhi.clear();
      pfTauChargedIso.clear();
      pfTauType.clear();
      pfTauIsoFlag.clear();
      pfTauRelIsoFlag.clear();
      pfTauPassesMediumIso.clear();

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

    unsigned short int nGlobalMuons;
    std::vector<float>   globalMuonPt;
    std::vector<float>   globalMuonEta;
    std::vector<float>   globalMuonPhi;
    std::vector<float>   globalMuonEtaAtVtx;
    std::vector<float>   globalMuonPhiAtVtx;
    std::vector<short int>   globalMuonIEt;
    std::vector<short int>   globalMuonIEta;
    std::vector<short int>   globalMuonIPhi;
    std::vector<short int>   globalMuonIEtaAtVtx;
    std::vector<short int>   globalMuonIPhiAtVtx;
    std::vector<short int>   globalMuonIDEta;
    std::vector<short int>   globalMuonIDPhi;
    std::vector<short int>      globalMuonChg;
    std::vector<unsigned short int> globalMuonIso;
    std::vector<unsigned short int> globalMuonQual;
    std::vector<unsigned short int> globalMuonTfMuonIdx;
    std::vector<short int>      globalMuonBx;


    unsigned short int nStandaloneMuons;
    std::vector<float>   standaloneMuonPt;
    std::vector<float>   standaloneMuonPt2;
    std::vector<float>   standaloneMuonEta;
    std::vector<float>   standaloneMuonPhi;
    std::vector<short int>      standaloneMuonChg;
    std::vector<unsigned short int> standaloneMuonQual;
    std::vector<short int>      standaloneMuonBx;
    std::vector<unsigned int>     standaloneMuonRegion;


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
    std::vector<bool> EGHGC;
    std::vector<bool>   EGPassesID;

    unsigned int nTkElectrons;
    std::vector<double> tkElectronEt;
    std::vector<double> tkElectronEta;
    std::vector<double> tkElectronPhi;
    std::vector<int>    tkElectronChg;
    std::vector<int>    tkElectronBx;
    std::vector<double> tkElectronTrkIso;
    std::vector<double> tkElectronzVtx;
    std::vector<double> tkElectronHwQual;
    std::vector<double>   tkElectronEGRefPt;
    std::vector<double>   tkElectronEGRefEta;
    std::vector<double>   tkElectronEGRefPhi;
    std::vector<bool> tkElectronHGC;
    std::vector<bool> tkElectronPassesID;

    unsigned int nTkElectronsLoose;
    std::vector<double> tkElectronLooseEt;
    std::vector<double> tkElectronLooseEta;
    std::vector<double> tkElectronLoosePhi;
    std::vector<double> tkElectronLooseChg;
    std::vector<int>    tkElectronLooseBx;
    std::vector<double> tkElectronLooseTrkIso;
    std::vector<double> tkElectronLoosezVtx;
    std::vector<double> tkElectronLooseHwQual;
    std::vector<double>   tkElectronLooseEGRefPt;
    std::vector<double>   tkElectronLooseEGRefEta;
    std::vector<double>   tkElectronLooseEGRefPhi;
    std::vector<bool> tkElectronLooseHGC;
    std::vector<bool> tkElectronLoosePassesID;

    unsigned int nTkPhotons;
    std::vector<double> tkPhotonEt;
    std::vector<double> tkPhotonEta;
    std::vector<double> tkPhotonPhi;
    std::vector<int>    tkPhotonBx;
    std::vector<double> tkPhotonTrkIso;
    std::vector<double> tkPhotonzVtx;
    std::vector<double> tkPhotonHwQual;
    std::vector<double>   tkPhotonEGRefPt;
    std::vector<double>   tkPhotonEGRefEta;
    std::vector<double>   tkPhotonEGRefPhi;
    std::vector<bool> tkPhotonHGC;
    std::vector<bool> tkPhotonPassesID;


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
    std::vector<double>   tkGlbMuonPt;
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
    std::vector<double>   tkMuonPt;
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
    std::vector<int>  tkMuonMuRefChg;
    std::vector<unsigned int>   tkMuonRegion;

    unsigned int nTkMuonStubs;
    std::vector<double>   tkMuonStubsPt;
    std::vector<double>   tkMuonStubsEta;
    std::vector<double>   tkMuonStubsPhi;
    std::vector<int>      tkMuonStubsChg;
    std::vector<unsigned int> tkMuonStubsIso;
    std::vector<double> tkMuonStubsTrkIso;
    std::vector<unsigned int> tkMuonStubsFwd;
    std::vector<unsigned int> tkMuonStubsMip;
    std::vector<unsigned int> tkMuonStubsRPC;
    std::vector<int>      tkMuonStubsBx;
    std::vector<unsigned int>      tkMuonStubsQuality;
    std::vector<double>   tkMuonStubszVtx;
    std::vector<double>   tkMuonStubsBarrelStubs;
    std::vector<unsigned int>   tkMuonStubsRegion;

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

    unsigned int  nPFMuons;
    std::vector<double> pfMuonPt;
    std::vector<double> pfMuonEta;
    std::vector<double> pfMuonPhi;
    std::vector<double>pfMuonzVtx;
    std::vector<int> pfMuonChg;

    unsigned int  nPFCands;
    std::vector<int> pfCandId;
    std::vector<double> pfCandEt;
    std::vector<double> pfCandEta;
    std::vector<double> pfCandPhi;
    std::vector<double>pfCandzVtx;
    std::vector<int> pfCandChg;

    unsigned int  nPFTaus;
    std::vector<double> pfTauEt;
    std::vector<double> pfTauEta;
    std::vector<double> pfTauPhi;
    std::vector<double> pfTauType;
    std::vector<double> pfTauChargedIso;
    std::vector<unsigned int> pfTauIsoFlag;
    std::vector<unsigned int> pfTauRelIsoFlag;
    std::vector<bool> pfTauPassesMediumIso;
    std::vector<int> pfTauChg;


  }; 
}
#endif


