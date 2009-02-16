//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Sep  1 19:12:16 2008 by ROOT version 5.18/00a
// from TTree OHltTree/
// found on file: TEST.root
//////////////////////////////////////////////////////////

#ifndef OHltTree_h
#define OHltTree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TRandom3.h>

#include <vector>
#include <string>
#include <map>

#include "OHltConfig.h"
#include "OHltMenu.h"
#include "OHltRateCounter.h"

class OHltTree {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           NrecoJetCal;
  Int_t           NrecoJetGen;
  Int_t           NrecoTowCal;
  Int_t           NrecoJetCorCal;
  Float_t         recoJetCalPt[43000];   //[NrecoJetCal]
  Float_t         recoJetCalPhi[43000];   //[NrecoJetCal]
  Float_t         recoJetCalEta[43000];   //[NrecoJetCal]
  Float_t         recoJetCalEt[43000];   //[NrecoJetCal]
  Float_t         recoJetCalE[43000];   //[NrecoJetCal]
  Float_t         recoJetGenPt[68000];   //[NrecoJetGen]
  Float_t         recoJetGenPhi[68000];   //[NrecoJetGen]
  Float_t         recoJetGenEta[68000];   //[NrecoJetGen]
  Float_t         recoJetGenEt[68000];   //[NrecoJetGen]
  Float_t         recoJetGenE[68000];   //[NrecoJetGen]
  Float_t         recoTowEt[684000];   //[NrecoTowCal]
  Float_t         recoTowEta[684000];   //[NrecoTowCal]
  Float_t         recoTowPhi[684000];   //[NrecoTowCal]
  Float_t         recoTowE[684000];   //[NrecoTowCal]
  Float_t         recoTowEm[684000];   //[NrecoTowCal]
  Float_t         recoTowHad[684000];   //[NrecoTowCal]
  Float_t         recoTowOE[684000];   //[NrecoTowCal]
  Float_t         recoJetCorCalPt[43000];   //[NrecoJetCorCal]
  Float_t         recoJetCorCalPhi[43000];   //[NrecoJetCorCal]
  Float_t         recoJetCorCalEta[43000];   //[NrecoJetCorCal]
  Float_t         recoJetCorCalE[43000];   //[NrecoJetCorCal]
  Float_t         recoMetCal;
  Float_t         recoMetCalPhi;
  Float_t         recoMetCalSum;
  Float_t         recoMetGen;
  Float_t         recoMetGenPhi;
  Float_t         recoMetGenSum;
  Float_t         recoHTCal;
  Float_t         recoHTCalPhi;
  Float_t         recoHTCalSum;
  Int_t           NohTau;
  Float_t         ohTauEta[5000];   //[NohTau]
  Float_t         ohTauPhi[5000];   //[NohTau]
  Float_t         ohTauPt[5000];   //[NohTau]
  Float_t         ohTauEiso[5000];   //[NohTau]
  Float_t         ohTauL25Tpt[5000];   //[NohTau]
  Int_t           ohTauL25Tiso[5000];   //[NohTau]
  Float_t         ohTauL3Tpt[5000];   //[NohTau]
  Int_t           ohTauL3Tiso[5000];   //[NohTau]
  Int_t           NohBJetL2;     //
  Float_t         ohBJetL2CorrectedEt[5000];  //[NohBJetL2]
  Float_t         ohBJetL2Et[5000];  //[NohBJetL2] 
  Int_t           NohBJetLife;
  Float_t         ohBJetLifeL2E[1000];   //[NohBJetLife]
  Float_t         ohBJetLifeL2ET[1000];   //[NohBJetLife]
  Float_t         ohBJetLifeL2Eta[1000];   //[NohBJetLife]
  Float_t         ohBJetLifeL2Phi[1000];   //[NohBJetLife]
  Float_t         ohBJetLifeL25Discriminator[1000];   //[NohBJetLife]
  Float_t         ohBJetLifeL3Discriminator[1000];   //[NohBJetLife]
  Int_t           NohBJetPixelTracks;
  Float_t         ohBJetLifePixelTrackPt[10000];   //[NohBJetPixelTracks]
  Float_t         ohBJetLifePixelTrackEta[10000];   //[NohBJetPixelTracks]
  Float_t         ohBJetLifePixelTrackPhi[10000];   //[NohBJetPixelTracks]
  Float_t         ohBJetLifePixelTrackChi2[10000];   //[NohBJetPixelTracks]
  Int_t           NohBJetRegionalTracks;
  Float_t         ohBJetLifeRegionalTrackPt[10000];   //[NohBJetRegionalTracks]
  Float_t         ohBJetLifeRegionalTrackEta[10000];   //[NohBJetRegionalTracks]
  Float_t         ohBJetLifeRegionalTrackPhi[10000];   //[NohBJetRegionalTracks]
  Float_t         ohBJetLifeRegionalTrackChi2[10000];   //[NohBJetRegionalTracks]
  Float_t         ohBJetLifeRegionalSeedPt[10000];   //[NohBJetRegionalTracks]
  Float_t         ohBJetLifeRegionalSeedEta[10000];   //[NohBJetRegionalTracks]
  Float_t         ohBJetLifeRegionalSeedPhi[10000];   //[NohBJetRegionalTracks]
  Int_t           NohBJetSoftm;
  Float_t         ohBJetSoftmL2E[1000];   //[NohBJetSoftm]
  Float_t         ohBJetSoftmL2ET[1000];   //[NohBJetSoftm]
  Float_t         ohBJetSoftmL2Eta[1000];   //[NohBJetSoftm]
  Float_t         ohBJetSoftmL2Phi[1000];   //[NohBJetSoftm]
  Int_t           ohBJetSoftmL25Discriminator[1000];   //[NohBJetSoftm]
  Float_t         ohBJetSoftmL3Discriminator[1000];   //[NohBJetSoftm]
  Int_t           NohBJetL2Corrected;
  Float_t         ohBJetPerfL2E[1000];   //[NohBJetL2Corrected]
  Float_t         ohBJetPerfL2ET[1000];   //[NohBJetL2Corrected]
  Float_t         ohBJetPerfL2Eta[1000];   //[NohBJetL2Corrected]
  Float_t         ohBJetPerfL2Phi[1000];   //[NohBJetL2Corrected]
  Int_t           ohBJetPerfL25Tag[1000];   //[NohBJetL2Corrected]
  Int_t           ohBJetPerfL3Tag[1000];   //[NohBJetL2Corrected]
  Int_t           NrecoElec;
  Float_t         recoElecPt[4000];   //[NrecoElec]
  Float_t         recoElecPhi[4000];   //[NrecoElec]
  Float_t         recoElecEta[4000];   //[NrecoElec]
  Float_t         recoElecEt[4000];   //[NrecoElec]
  Float_t         recoElecE[4000];   //[NrecoElec]
  Int_t           NrecoPhot;
  Float_t         recoPhotPt[5000];   //[NrecoPhot]
  Float_t         recoPhotPhi[5000];   //[NrecoPhot]
  Float_t         recoPhotEta[5000];   //[NrecoPhot]
  Float_t         recoPhotEt[5000];   //[NrecoPhot]
  Float_t         recoPhotE[5000];   //[NrecoPhot]
  Int_t           NohPhot;
  Float_t         ohPhotEt[8000];   //[NohPhot]
  Float_t         ohPhotEta[8000];   //[NohPhot]
  Float_t         ohPhotPhi[8000];   //[NohPhot]
  Float_t         ohPhotEiso[8000];   //[NohPhot]
  Float_t         ohPhotHiso[8000];   //[NohPhot]
  Float_t         ohPhotTiso[8000];   //[NohPhot]
  Int_t           ohPhotL1iso[8000];   //[NohPhot]
  Int_t           NohEle;
  Float_t         ohEleEt[8000];   //[NohEle]
  Float_t         ohEleEta[8000];   //[NohEle]
  Float_t         ohElePhi[8000];   //[NohEle]
  Float_t         ohEleE[8000];   //[NohEle]
  Float_t         ohEleP[8000];   //[NohEle]
  Float_t         ohEleHiso[8000];   //[NohEle]
  Float_t         ohEleTiso[8000];   //[NohEle]
  Int_t           ohEleL1iso[8000];   //[NohEle]
  Int_t           ohElePixelSeeds[8000];   //[NohEle]
  Int_t           ohEleNewSC[8000];   //[NohEle]
  Int_t           NohEleLW;
  Float_t         ohEleEtLW[11000];   //[NohEleLW]
  Float_t         ohEleEtaLW[11000];   //[NohEleLW]
  Float_t         ohElePhiLW[11000];   //[NohEleLW]
  Float_t         ohEleELW[11000];   //[NohEleLW]
  Float_t         ohElePLW[11000];   //[NohEleLW]
  Float_t         ohEleHisoLW[11000];   //[NohEleLW]
  Float_t         ohEleTisoLW[11000];   //[NohEleLW]
  Int_t           ohEleL1isoLW[11000];   //[NohEleLW]
  Int_t           ohElePixelSeedsLW[11000];   //[NohEleLW]
  Int_t           ohEleNewSCLW[11000];   //[NohEleLW]
  Int_t           NrecoMuon;
  Float_t         recoMuonPt[5000];   //[NrecoMuon]
  Float_t         recoMuonPhi[5000];   //[NrecoMuon]
  Float_t         recoMuonEta[5000];   //[NrecoMuon]
  Float_t         recoMuonEt[5000];   //[NrecoMuon]
  Float_t         recoMuonE[5000];   //[NrecoMuon]
  Int_t           NohMuL2;
  Float_t         ohMuL2Pt[2000];   //[NohMuL2]
  Float_t         ohMuL2Phi[2000];   //[NohMuL2]
  Float_t         ohMuL2Eta[2000];   //[NohMuL2]
  Int_t           ohMuL2Chg[2000];   //[NohMuL2]
  Float_t         ohMuL2PtErr[2000];   //[NohMuL2]
  Int_t           ohMuL2Iso[2000];   //[NohMuL2]
  Float_t         ohMuL2Dr[2000];   //[NohMuL2]
  Float_t         ohMuL2Dz[2000];   //[NohMuL2]
  Int_t           NohMuL3;
  Float_t         ohMuL3Pt[1000];   //[NohMuL3]
  Float_t         ohMuL3Phi[1000];   //[NohMuL3]
  Float_t         ohMuL3Eta[1000];   //[NohMuL3]
  Int_t           ohMuL3Chg[1000];   //[NohMuL3]
  Float_t         ohMuL3PtErr[1000];   //[NohMuL3]
  Int_t           ohMuL3Iso[1000];   //[NohMuL3]
  Float_t         ohMuL3Dr[1000];   //[NohMuL3]
  Float_t         ohMuL3Dz[1000];   //[NohMuL3]
  Int_t           ohMuL3L2idx[1000];   //[NohMuL3]
  Int_t           NMCpart;
  Int_t           MCpid[1203000];   //[NMCpart]
  Int_t           MCstatus[1203000];   //[NMCpart]
  Float_t         MCvtxX[1203000];   //[NMCpart]
  Float_t         MCvtxY[1203000];   //[NMCpart]
  Float_t         MCvtxZ[1203000];   //[NMCpart]
  Float_t         MCpt[1203000];   //[NMCpart]
  Float_t         MCeta[1203000];   //[NMCpart]
  Float_t         MCphi[1203000];   //[NMCpart]
  Float_t         MCPtHat;
  Int_t           MCmu3;
  Int_t           MCel3;
  Int_t           MCbb;
  Int_t           MCab;
  Int_t           MCWenu;
  Int_t           MCWmunu;
  Int_t           MCZee;
  Int_t           MCZmumu;
  Float_t         MCptEleMax;
  Float_t         MCptMuMax;
  Int_t           NL1IsolEm;
  Float_t         L1IsolEmEt[4000];   //[NL1IsolEm]
  Float_t         L1IsolEmE[4000];   //[NL1IsolEm]
  Float_t         L1IsolEmEta[4000];   //[NL1IsolEm]
  Float_t         L1IsolEmPhi[4000];   //[NL1IsolEm]
  Int_t           NL1NIsolEm;
  Float_t         L1NIsolEmEt[4000];   //[NL1NIsolEm]
  Float_t         L1NIsolEmE[4000];   //[NL1NIsolEm]
  Float_t         L1NIsolEmEta[4000];   //[NL1NIsolEm]
  Float_t         L1NIsolEmPhi[4000];   //[NL1NIsolEm]
  Int_t           NL1Mu;
  Float_t         L1MuPt[4000];   //[NL1Mu]
  Float_t         L1MuE[4000];   //[NL1Mu]
  Float_t         L1MuEta[4000];   //[NL1Mu]
  Float_t         L1MuPhi[4000];   //[NL1Mu]
  Int_t           L1MuIsol[4000];   //[NL1Mu]
  Int_t           L1MuMip[4000];   //[NL1Mu]
  Int_t           L1MuFor[4000];   //[NL1Mu]
  Int_t           L1MuRPC[4000];   //[NL1Mu]
  Int_t           L1MuQal[4000];   //[NL1Mu]
  Int_t           NL1CenJet;
  Float_t         L1CenJetEt[4000];   //[NL1CenJet]
  Float_t         L1CenJetE[4000];   //[NL1CenJet]
  Float_t         L1CenJetEta[4000];   //[NL1CenJet]
  Float_t         L1CenJetPhi[4000];   //[NL1CenJet]
  Int_t           NL1ForJet;
  Float_t         L1ForJetEt[4000];   //[NL1ForJet]
  Float_t         L1ForJetE[4000];   //[NL1ForJet]
  Float_t         L1ForJetEta[4000];   //[NL1ForJet]
  Float_t         L1ForJetPhi[4000];   //[NL1ForJet]
  Int_t           NL1Tau;
  Float_t         L1TauEt[4000];   //[NL1Tau]
  Float_t         L1TauE[4000];   //[NL1Tau]
  Float_t         L1TauEta[4000];   //[NL1Tau]
  Float_t         L1TauPhi[4000];   //[NL1Tau]
  Float_t         L1Met;
  Float_t         L1MetPhi;
  Float_t         L1MetTot;
  Float_t         L1MetHad;
  Int_t           L1HfRing0EtSumPositiveEta;
  Int_t           L1HfRing1EtSumPositiveEta;
  Int_t           L1HfRing0EtSumNegativeEta;
  Int_t           L1HfRing1EtSumNegativeEta;
  Int_t           L1HfTowerCountPositiveEta;
  Int_t           L1HfTowerCountNegativeEta;
  Int_t           Run;
  Int_t           Event;

  //L1's
  Int_t           L1_DoubleEG10_00001; 
  Int_t           L1_DoubleEG1_00001; 
  Int_t           L1_DoubleEG5_00001; 
  Int_t           L1_DoubleForJet20; 
  Int_t           L1_DoubleHfBitCountsRing1_P1N1; 
  Int_t           L1_DoubleHfBitCountsRing2_P1N1; 
  Int_t           L1_DoubleHfRingEtSumsRing1_P200N200; 
  Int_t           L1_DoubleHfRingEtSumsRing1_P4N4; 
  Int_t           L1_DoubleHfRingEtSumsRing2_P200N200; 
  Int_t           L1_DoubleHfRingEtSumsRing2_P4N4; 
  Int_t           L1_DoubleIsoEG05_TopBottom; 
  Int_t           L1_DoubleIsoEG05_TopBottomCen; 
  Int_t           L1_DoubleIsoEG10_00001; 
  Int_t           L1_DoubleIsoEG8_00001; 
  Int_t           L1_DoubleJet40_00001; 
  Int_t           L1_DoubleJet60_00001; 
  Int_t           L1_DoubleMu3; 
  Int_t           L1_DoubleMuOpen; 
  Int_t           L1_DoubleMuTopBottom; 
  Int_t           L1_DoubleNoIsoEG05_TopBottom; 
  Int_t           L1_DoubleNoIsoEG05_TopBottomCen; 
  Int_t           L1_DoubleTauJet20_00001; 
  Int_t           L1_DoubleTauJet8_00001; 
  Int_t           L1_EG12_Jet40_00001; 
  Int_t           L1_EG5_TripleJet6_00001; 
  Int_t           L1_ETM20_00001; 
  Int_t           L1_ETM30_00001; 
  Int_t           L1_ETM40_00001; 
  Int_t           L1_ETM50_00001; 
  Int_t           L1_ETT60_00001; 
  Int_t           L1_HTT100_00001; 
  Int_t           L1_HTT200_00001; 
  Int_t           L1_HTT300_00001; 
  Int_t           L1_IsoEG10_Jet12_00001; 
  Int_t           L1_IsoEG10_Jet6_00001; 
  Int_t           L1_IsoEG10_Jet6_ForJet6_00001; 
  Int_t           L1_IsoEG10_Jet8_00001; 
  Int_t           L1_IsoEG10_TauJet8_00001; 
  Int_t           L1_MinBias_ETT10_00001; 
  Int_t           L1_MinBias_HTT10_00001; 
  Int_t           L1_Mu3_EG12_00001; 
  Int_t           L1_Mu3_IsoEG5_00001; 
  Int_t           L1_Mu3_Jet6_00001; 
  Int_t           L1_Mu3_TripleJet6_00001; 
  Int_t           L1_Mu5_IsoEG10_00001; 
  Int_t           L1_Mu5_Jet6_00001; 
  Int_t           L1_Mu5_TauJet8_00001; 
  Int_t           L1_QuadJet20_00001; 
  Int_t           L1_QuadJet6_00001; 
  Int_t           L1_SingleEG1; 
  Int_t           L1_SingleEG10_00001; 
  Int_t           L1_SingleEG12_00001; 
  Int_t           L1_SingleEG15_00001; 
  Int_t           L1_SingleEG20_00001; 
  Int_t           L1_SingleEG5_00001; 
  Int_t           L1_SingleEG5_Endcap_00001; 
  Int_t           L1_SingleEG8_00001; 
  Int_t           L1_SingleForJet10; 
  Int_t           L1_SingleForJet6; 
  Int_t           L1_SingleHfBitCountsRing1_1; 
  Int_t           L1_SingleHfBitCountsRing2_1; 
  Int_t           L1_SingleHfRingEtSumsRing1_200; 
  Int_t           L1_SingleHfRingEtSumsRing1_4; 
  Int_t           L1_SingleHfRingEtSumsRing2_200; 
  Int_t           L1_SingleHfRingEtSumsRing2_4; 
  Int_t           L1_SingleIsoEG10_00001; 
  Int_t           L1_SingleIsoEG12_00001; 
  Int_t           L1_SingleIsoEG15_00001; 
  Int_t           L1_SingleIsoEG5_00001; 
  Int_t           L1_SingleIsoEG5_Endcap_00001; 
  Int_t           L1_SingleIsoEG8_00001; 
  Int_t           L1_SingleJet10_00001; 
  Int_t           L1_SingleJet10_Barrel_00001; 
  Int_t           L1_SingleJet10_Central; 
  Int_t           L1_SingleJet10_Endcap; 
  Int_t           L1_SingleJet20_00001; 
  Int_t           L1_SingleJet20_Barrel_00001; 
  Int_t           L1_SingleJet30_00001; 
  Int_t           L1_SingleJet30_Barrel_00001; 
  Int_t           L1_SingleJet40_00001; 
  Int_t           L1_SingleJet40_Barrel_00001; 
  Int_t           L1_SingleJet50_00001; 
  Int_t           L1_SingleJet60_00001; 
  Int_t           L1_SingleJet6_00001; 
  Int_t           L1_SingleJet6_Barrel_00001; 
  Int_t           L1_SingleJet6_Central; 
  Int_t           L1_SingleJet6_Endcap; 
  Int_t           L1_SingleMu0; 
  Int_t           L1_SingleMu10; 
  Int_t           L1_SingleMu14; 
  Int_t           L1_SingleMu3; 
  Int_t           L1_SingleMu5; 
  Int_t           L1_SingleMu7; 
  Int_t           L1_SingleMuBeamHalo; 
  Int_t           L1_SingleMuOpen; 
  Int_t           L1_SingleTauJet10_00001; 
  Int_t           L1_SingleTauJet10_Barrel_00001; 
  Int_t           L1_SingleTauJet20_00001; 
  Int_t           L1_SingleTauJet20_Barrel_00001; 
  Int_t           L1_SingleTauJet30_00001; 
  Int_t           L1_SingleTauJet30_Barrel_00001; 
  Int_t           L1_SingleTauJet50_00001; 
  Int_t           L1_SingleTauJet8_00001; 
  Int_t           L1_SingleTauJet8_Barrel_00001; 
  Int_t           L1_TauJet10_ETM30_00001; 
  Int_t           L1_TauJet10_ETM40_00001; 
  Int_t           L1_TripleJet30_00001; 
  Int_t           L1_TripleMu3; 

  // Here we declare any emulated L1 bits 
  Int_t           OpenL1_ZeroBias;
  Int_t           OpenL1_Mu3EG5; 

  Int_t           HLT_L1Jet15;
  Int_t           HLT_Jet30;
  Int_t           HLT_Jet50;
  Int_t           HLT_Jet80;
  Int_t           HLT_Jet110;
  Int_t           HLT_Jet180;
  Int_t           HLT_Jet250;
  Int_t           HLT_FwdJet20;
  Int_t           HLT_DoubleJet150;
  Int_t           HLT_DoubleJet125_Aco;
  Int_t           HLT_DoubleFwdJet50;
  Int_t           HLT_DiJetAve15;
  Int_t           HLT_DiJetAve30;
  Int_t           HLT_DiJetAve50;
  Int_t           HLT_DiJetAve70;
  Int_t           HLT_DiJetAve130;
  Int_t           HLT_DiJetAve220;
  Int_t           HLT_TripleJet85;
  Int_t           HLT_QuadJet30;
  Int_t           HLT_QuadJet60;
  Int_t           HLT_SumET120;
  Int_t           HLT_L1MET20;
  Int_t           HLT_MET25;
  Int_t           HLT_MET35;
  Int_t           HLT_MET50;
  Int_t           HLT_MET65;
  Int_t           HLT_MET75;
  Int_t           HLT_MET35_HT350;
  Int_t           HLT_Jet180_MET60;
  Int_t           HLT_Jet60_MET70_Aco;
  Int_t           HLT_Jet100_MET60_Aco;
  Int_t           HLT_DoubleJet125_MET60;
  Int_t           HLT_DoubleFwdJet40_MET60;
  Int_t           HLT_DoubleJet60_MET60_Aco;
  Int_t           HLT_DoubleJet50_MET70_Aco;
  Int_t           HLT_DoubleJet40_MET70_Aco;
  Int_t           HLT_TripleJet60_MET60;
  Int_t           HLT_QuadJet35_MET60;
  Int_t           HLT_IsoEle15_L1I;
  Int_t           HLT_IsoEle18_L1R;
  Int_t           HLT_IsoEle15_LW_L1I;
  Int_t           HLT_LooseIsoEle15_LW_L1R;
  Int_t           HLT_Ele10_SW_L1R;
  Int_t           HLT_Ele15_SW_L1R;
  Int_t           HLT_Ele15_LW_L1R;
  Int_t           HLT_EM80;
  Int_t           HLT_EM200;
  Int_t           HLT_DoubleIsoEle10_L1I;
  Int_t           HLT_DoubleIsoEle12_L1R;
  Int_t           HLT_DoubleIsoEle10_LW_L1I;
  Int_t           HLT_DoubleIsoEle12_LW_L1R;
  Int_t           HLT_DoubleEle5_SW_L1R;
  Int_t           HLT_DoubleEle10_LW_OnlyPixelM_L1R;
  Int_t           HLT_DoubleEle10_Z;
  Int_t           HLT_DoubleEle6_Exclusive;
  Int_t           HLT_IsoPhoton30_L1I;
  Int_t           HLT_IsoPhoton10_L1R;
  Int_t           HLT_IsoPhoton15_L1R;
  Int_t           HLT_IsoPhoton20_L1R;
  Int_t           HLT_IsoPhoton25_L1R;
  Int_t           HLT_IsoPhoton40_L1R;
  Int_t           HLT_Photon15_L1R;
  Int_t           HLT_Photon25_L1R;
  Int_t           HLT_DoubleIsoPhoton20_L1I;
  Int_t           HLT_DoubleIsoPhoton20_L1R;
  Int_t           HLT_DoublePhoton10_Exclusive;
  Int_t           HLT_L1Mu;
  Int_t           HLT_L1MuOpen;
  Int_t           HLT_L2Mu9;
  Int_t           HLT_IsoMu9;
  Int_t           HLT_IsoMu11;
  Int_t           HLT_IsoMu13;
  Int_t           HLT_IsoMu15;
  Int_t           HLT_NoTrackerIsoMu15;
  Int_t           HLT_Mu3;
  Int_t           HLT_Mu5;
  Int_t           HLT_Mu7;
  Int_t           HLT_Mu9;
  Int_t           HLT_Mu11;
  Int_t           HLT_Mu13;
  Int_t           HLT_Mu15;
  Int_t           HLT_Mu15_L1Mu7;
  Int_t           HLT_Mu15_Vtx2cm;
  Int_t           HLT_Mu15_Vtx2mm;
  Int_t           HLT_DoubleIsoMu3;
  Int_t           HLT_DoubleMu3;
  Int_t           HLT_DoubleMu3_Vtx2cm;
  Int_t           HLT_DoubleMu3_Vtx2mm;
  Int_t           HLT_DoubleMu3_JPsi;
  Int_t           HLT_DoubleMu3_Upsilon;
  Int_t           HLT_DoubleMu7_Z;
  Int_t           HLT_DoubleMu3_SameSign;
  Int_t           HLT_DoubleMu3_Psi2S;
  Int_t           HLT_BTagIP_Jet180;
  Int_t           HLT_BTagIP_Jet120_Relaxed;
  Int_t           HLT_BTagIP_DoubleJet120;
  Int_t           HLT_BTagIP_DoubleJet60_Relaxed;
  Int_t           HLT_BTagIP_TripleJet70;
  Int_t           HLT_BTagIP_TripleJet40_Relaxed;
  Int_t           HLT_BTagIP_QuadJet40;
  Int_t           HLT_BTagIP_QuadJet30_Relaxed;
  Int_t           HLT_BTagIP_HT470;
  Int_t           HLT_BTagIP_HT320_Relaxed;
  Int_t           HLT_BTagMu_DoubleJet120;
  Int_t           HLT_BTagMu_DoubleJet60_Relaxed;
  Int_t           HLT_BTagMu_TripleJet70;
  Int_t           HLT_BTagMu_TripleJet40_Relaxed;
  Int_t           HLT_BTagMu_QuadJet40;
  Int_t           HLT_BTagMu_QuadJet30_Relaxed;
  Int_t           HLT_BTagMu_HT370;
  Int_t           HLT_BTagMu_HT250_Relaxed;
  Int_t           HLT_DoubleMu3_BJPsi;
  Int_t           HLT_DoubleMu4_BJPsi;
  Int_t           HLT_TripleMu3_TauTo3Mu;
  Int_t           HLT_IsoTau_MET65_Trk20;
  Int_t           HLT_IsoTau_MET35_Trk15_L1MET;
  Int_t           HLT_LooseIsoTau_MET30;
  Int_t           HLT_LooseIsoTau_MET30_L1MET;
  Int_t           HLT_DoubleIsoTau_Trk3;
  Int_t           HLT_DoubleLooseIsoTau;
  Int_t           HLT_IsoEle8_IsoMu7;
  Int_t           HLT_IsoEle10_Mu10_L1R;
  Int_t           HLT_IsoEle12_IsoTau_Trk3;
  Int_t           HLT_IsoEle10_BTagIP_Jet35;
  Int_t           HLT_IsoEle12_Jet40;
  Int_t           HLT_IsoEle12_DoubleJet80;
  Int_t           HLT_IsoElec5_TripleJet30;
  Int_t           HLT_IsoEle12_TripleJet60;
  Int_t           HLT_IsoEle12_QuadJet35;
  Int_t           HLT_IsoMu14_IsoTau_Trk3;
  Int_t           HLT_IsoMu7_BTagIP_Jet35;
  Int_t           HLT_IsoMu7_BTagMu_Jet20;
  Int_t           HLT_IsoMu7_Jet40;
  Int_t           HLT_NoL2IsoMu8_Jet40;
  Int_t           HLT_Mu14_Jet50;
  Int_t           HLT_Mu5_TripleJet30;
  Int_t           HLT_BTagMu_Jet20_Calib;
  Int_t           HLT_ZeroBias;
  Int_t           HLT_MinBias;
  Int_t           HLT_MinBiasHcal;
  Int_t           HLT_MinBiasEcal;
  Int_t           HLT_MinBiasPixel;
  Int_t           HLT_MinBiasPixel_Trk5;
  Int_t           HLT_BackwardBSC;
  Int_t           HLT_ForwardBSC;
  Int_t           HLT_CSCBeamHalo;
  Int_t           HLT_CSCBeamHaloOverlapRing1;
  Int_t           HLT_CSCBeamHaloOverlapRing2;
  Int_t           HLT_CSCBeamHaloRing2or3;
  Int_t           HLT_TrackerCosmics;
  Int_t           HLT_TriggerType;
  Int_t           AlCa_IsoTrack;
  Int_t           AlCa_EcalPhiSym;
  Int_t           AlCa_EcalPi0;
  Int_t           HLTriggerFinalPath;

  // List of branches
  TBranch        *b_NrecoJetCal;   //!
  TBranch        *b_NrecoJetGen;   //!
  TBranch        *b_NrecoTowCal;   //!
  TBranch        *b_NrecoJetCorCal; //!
  TBranch        *b_recoJetCalPt;   //!
  TBranch        *b_recoJetCalPhi;   //!
  TBranch        *b_recoJetCalEta;   //!
  TBranch        *b_recoJetCalEt;   //!
  TBranch        *b_recoJetCalE;   //!
  TBranch        *b_recoJetGenPt;   //!
  TBranch        *b_recoJetGenPhi;   //!
  TBranch        *b_recoJetGenEta;   //!
  TBranch        *b_recoJetGenEt;   //!
  TBranch        *b_recoJetGenE;   //!
  TBranch        *b_recoTowEt;   //!
  TBranch        *b_recoTowEta;   //!
  TBranch        *b_recoTowPhi;   //!
  TBranch        *b_recoTowE;   //!
  TBranch        *b_recoTowEm;   //!
  TBranch        *b_recoTowHad;   //!
  TBranch        *b_recoTowOE;   //!
  TBranch        *b_recoMetCal;   //!
  TBranch        *b_recoMetCalPhi;   //!
  TBranch        *b_recoMetCalSum;   //!
  TBranch        *b_recoMetGen;   //!
  TBranch        *b_recoMetGenPhi;   //!
  TBranch        *b_recoMetGenSum;   //!
  TBranch        *b_recoHTCal;   //!
  TBranch        *b_recoHTCalPhi;   //!
  TBranch        *b_recoHTCalSum;   //!
  TBranch        *b_recoJetCorCalPt;   //!
  TBranch        *b_recoJetCorCalPhi;   //!
  TBranch        *b_recoJetCorCalEta;   //!
  TBranch        *b_recoJetCorCalE;   //!
  TBranch        *b_NohTau;   //!
  TBranch        *b_ohTauEta;   //!
  TBranch        *b_ohTauPhi;   //!
  TBranch        *b_ohTauPt;   //!
  TBranch        *b_ohTauEiso;   //!
  TBranch        *b_ohTauL25Tpt;   //!
  TBranch        *b_ohTauL25Tiso;   //!
  TBranch        *b_ohTauL3Tpt;   //!
  TBranch        *b_ohTauL3Tiso;   //!
  TBranch        *b_NohBJetL2;      //!
  TBranch        *b_ohBJetL2CorrectedEt;    //!
  TBranch        *b_ohBJetL2Et;    //!
  TBranch        *b_NohBJetLife;   //!
  TBranch        *b_ohBJetLifeL2E;   //!
  TBranch        *b_ohBJetLifeL2ET;   //!
  TBranch        *b_ohBJetLifeL2Eta;   //!
  TBranch        *b_ohBJetLifeL2Phi;   //!
  TBranch        *b_ohBJetLifeL25Discriminator;   //!
  TBranch        *b_ohBJetLifeL3Discriminator;   //!
  TBranch        *b_NohBJetPixelTracks;   //!
  TBranch        *b_ohBJetLifePixelTrackPt;   //!
  TBranch        *b_ohBJetLifePixelTrackEta;   //!
  TBranch        *b_ohBJetLifePixelTrackPhi;   //!
  TBranch        *b_ohBJetLifePixelTrackChi2;   //!
  TBranch        *b_NohBJetRegionalTracks;   //!
  TBranch        *b_ohBJetLifeRegionalTrackPt;   //!
  TBranch        *b_ohBJetLifeRegionalTrackEta;   //!
  TBranch        *b_ohBJetLifeRegionalTrackPhi;   //!
  TBranch        *b_ohBJetLifeRegionalTrackChi2;   //!
  TBranch        *b_ohBJetLifeRegionalSeedPt;   //!
  TBranch        *b_ohBJetLifeRegionalSeedEta;   //!
  TBranch        *b_ohBJetLifeRegionalSeedPhi;   //!
  TBranch        *b_NohBJetSoftm;   //!
  TBranch        *b_ohBJetSoftmL2E;   //!
  TBranch        *b_ohBJetSoftmL2ET;   //!
  TBranch        *b_ohBJetSoftmL2Eta;   //!
  TBranch        *b_ohBJetSoftmL2Phi;   //!
  TBranch        *b_ohBJetSoftmL25Discriminator;   //!
  TBranch        *b_ohBJetSoftmL3Discriminator;   //!
  TBranch        *b_NohBJetL2Corrected;   //!
  TBranch        *b_ohBJetPerfL2E;   //!
  TBranch        *b_ohBJetPerfL2ET;   //!
  TBranch        *b_ohBJetPerfL2Eta;   //!
  TBranch        *b_ohBJetPerfL2Phi;   //!
  TBranch        *b_ohBJetPerfL25Tag;   //!
  TBranch        *b_ohBJetPerfL3Tag;   //!
  TBranch        *b_NrecoElec;   //!
  TBranch        *b_recoElecPt;   //!
  TBranch        *b_recoElecPhi;   //!
  TBranch        *b_recoElecEta;   //!
  TBranch        *b_recoElecEt;   //!
  TBranch        *b_recoElecE;   //!
  TBranch        *b_NrecoPhot;   //!
  TBranch        *b_recoPhotPt;   //!
  TBranch        *b_recoPhotPhi;   //!
  TBranch        *b_recoPhotEta;   //!
  TBranch        *b_recoPhotEt;   //!
  TBranch        *b_recoPhotE;   //!
  TBranch        *b_NohPhot;   //!
  TBranch        *b_ohPhotEt;   //!
  TBranch        *b_ohPhotEta;   //!
  TBranch        *b_ohPhotPhi;   //!
  TBranch        *b_ohPhotEiso;   //!
  TBranch        *b_ohPhotHiso;   //!
  TBranch        *b_ohPhotTiso;   //!
  TBranch        *b_ohPhotL1iso;   //!
  TBranch        *b_NohEle;   //!
  TBranch        *b_ohEleEt;   //!
  TBranch        *b_ohEleEta;   //!
  TBranch        *b_ohElePhi;   //!
  TBranch        *b_ohEleE;   //!
  TBranch        *b_ohEleP;   //!
  TBranch        *b_ohEleHiso;   //!
  TBranch        *b_ohEleTiso;   //!
  TBranch        *b_ohEleL1iso;   //!
  TBranch        *b_ohElePixelSeeds;   //!
  TBranch        *b_ohEleNewSC;   //!
  TBranch        *b_NohEleLW;   //!
  TBranch        *b_ohEleEtLW;   //!
  TBranch        *b_ohEleEtaLW;   //!
  TBranch        *b_ohElePhiLW;   //!
  TBranch        *b_ohEleELW;   //!
  TBranch        *b_ohElePLW;   //!
  TBranch        *b_ohEleHisoLW;   //!
  TBranch        *b_ohEleTisoLW;   //!
  TBranch        *b_ohEleL1isoLW;   //!
  TBranch        *b_ohElePixelSeedsLW;   //!
  TBranch        *b_ohEleNewSCLW;   //!
  TBranch        *b_NrecoMuon;   //!
  TBranch        *b_recoMuonPt;   //!
  TBranch        *b_recoMuonPhi;   //!
  TBranch        *b_recoMuonEta;   //!
  TBranch        *b_recoMuonEt;   //!
  TBranch        *b_recoMuonE;   //!
  TBranch        *b_NohMuL2;   //!
  TBranch        *b_ohMuL2Pt;   //!
  TBranch        *b_ohMuL2Phi;   //!
  TBranch        *b_ohMuL2Eta;   //!
  TBranch        *b_ohMuL2Chg;   //!
  TBranch        *b_ohMuL2PtErr;   //!
  TBranch        *b_ohMuL2Iso;   //!
  TBranch        *b_ohMuL2Dr;   //!
  TBranch        *b_ohMuL2Dz;   //!
  TBranch        *b_NohMuL3;   //!
  TBranch        *b_ohMuL3Pt;   //!
  TBranch        *b_ohMuL3Phi;   //!
  TBranch        *b_ohMuL3Eta;   //!
  TBranch        *b_ohMuL3Chg;   //!
  TBranch        *b_ohMuL3PtErr;   //!
  TBranch        *b_ohMuL3Iso;   //!
  TBranch        *b_ohMuL3Dr;   //!
  TBranch        *b_ohMuL3Dz;   //!
  TBranch        *b_ohMuL3L2idx;   //!
  TBranch        *b_NMCpart;   //!
  TBranch        *b_MCpid;   //!
  TBranch        *b_MCstatus;   //!
  TBranch        *b_MCvtxX;   //!
  TBranch        *b_MCvtxY;   //!
  TBranch        *b_MCvtxZ;   //!
  TBranch        *b_MCpt;   //!
  TBranch        *b_MCeta;   //!
  TBranch        *b_MCphi;   //!
  TBranch        *b_MCPtHat;   //!
  TBranch        *b_MCmu3;   //!
  TBranch        *b_MCel3;   //!
  TBranch        *b_MCbb;   //!
  TBranch        *b_MCab;   //!
  TBranch        *b_MCWenu;   //!
  TBranch        *b_MCmunu;   //!
  TBranch        *b_MCZee;   //!
  TBranch        *b_MCZmumu;   //!
  TBranch        *b_MCptEleMax;   //!
  TBranch        *b_MCptMuMax;   //!
  TBranch        *b_NL1IsolEm;   //!
  TBranch        *b_L1IsolEmEt;   //!
  TBranch        *b_L1IsolEmE;   //!
  TBranch        *b_L1IsolEmEta;   //!
  TBranch        *b_L1IsolEmPhi;   //!
  TBranch        *b_NL1NIsolEm;   //!
  TBranch        *b_L1NIsolEmEt;   //!
  TBranch        *b_L1NIsolEmE;   //!
  TBranch        *b_L1NIsolEmEta;   //!
  TBranch        *b_L1NIsolEmPhi;   //!
  TBranch        *b_NL1Mu;   //!
  TBranch        *b_L1MuPt;   //!
  TBranch        *b_L1MuE;   //!
  TBranch        *b_L1MuEta;   //!
  TBranch        *b_L1MuPhi;   //!
  TBranch        *b_L1MuIsol;   //!
  TBranch        *b_L1MuMip;   //!
  TBranch        *b_L1MuFor;   //!
  TBranch        *b_L1MuRPC;   //!
  TBranch        *b_L1MuQal;   //!
  TBranch        *b_NL1CenJet;   //!
  TBranch        *b_L1CenJetEt;   //!
  TBranch        *b_L1CenJetE;   //!
  TBranch        *b_L1CenJetEta;   //!
  TBranch        *b_L1CenJetPhi;   //!
  TBranch        *b_NL1ForJet;   //!
  TBranch        *b_L1ForJetEt;   //!
  TBranch        *b_L1ForJetE;   //!
  TBranch        *b_L1ForJetEta;   //!
  TBranch        *b_L1ForJetPhi;   //!
  TBranch        *b_NL1Tau;   //!
  TBranch        *b_L1TauEt;   //!
  TBranch        *b_L1TauE;   //!
  TBranch        *b_L1TauEta;   //!
  TBranch        *b_L1TauPhi;   //!
  TBranch        *b_L1Met;   //!
  TBranch        *b_L1MetPhi;   //!
  TBranch        *b_L1MetTot;   //!
  TBranch        *b_L1MetHad;   //!
  TBranch        *b_L1HfRing0EtSumPositiveEta;   //!
  TBranch        *b_L1HfRing1EtSumPositiveEta;   //!
  TBranch        *b_L1HfRing0EtSumNegativeEta;   //!
  TBranch        *b_L1HfRing1EtSumNegativeEta;   //!
  TBranch        *b_L1HfTowerCountPositiveEta;   //!
  TBranch        *b_L1HfTowerCountNegativeEta;   //!
  TBranch        *b_Run;   //!
  TBranch        *b_Event;   //!
  TBranch        *b_HLT1jet;   //!
  TBranch        *b_HLT2jet;   //!
  TBranch        *b_HLT3jet;   //!
  TBranch        *b_HLT4jet;   //!
  TBranch        *b_HLT1MET;   //!
  TBranch        *b_HLT2jetAco;   //!
  TBranch        *b_HLT1jet1METAco;   //!
  TBranch        *b_HLT1jet1MET;   //!
  TBranch        *b_HLT2jet1MET;   //!
  TBranch        *b_HLT3jet1MET;   //!
  TBranch        *b_HLT4jet1MET;   //!
  TBranch        *b_HLT1MET1HT;   //!
  TBranch        *b_CandHLT1SumET;   //!
  TBranch        *b_HLT1jetPE1;   //!
  TBranch        *b_HLT1jetPE3;   //!
  TBranch        *b_HLT1jetPE5;   //!
  TBranch        *b_HLT1jetPE7;   //!
  TBranch        *b_HLT1METPre1;   //!
  TBranch        *b_HLT1METPre2;   //!
  TBranch        *b_HLT1METPre3;   //!
  TBranch        *b_HLT2jetAve30;   //!
  TBranch        *b_HLT2jetAve60;   //!
  TBranch        *b_HLT2jetAve110;   //!
  TBranch        *b_HLT2jetAve150;   //!
  TBranch        *b_HLT2jetAve200;   //!
  TBranch        *b_HLT2jetvbfMET;   //!
  TBranch        *b_HLTS2jet1METNV;   //!
  TBranch        *b_HLTS2jet1METAco;   //!
  TBranch        *b_HLTSjet1MET1Aco;   //!
  TBranch        *b_HLTSjet2MET1Aco;   //!
  TBranch        *b_HLTS2jetMET1Aco;   //!
  TBranch        *b_HLTJetMETRapidityGap;   //!
  TBranch        *b_HLT1Electron;   //!
  TBranch        *b_HLT1ElectronRelaxed;   //!
  TBranch        *b_HLT2Electron;   //!
  TBranch        *b_HLT2ElectronRelaxed;   //!
  TBranch        *b_HLT1Photon;   //!
  TBranch        *b_HLT1PhotonRelaxed;   //!
  TBranch        *b_HLT2Photon;   //!
  TBranch        *b_HLT2PhotonRelaxed;   //!
  TBranch        *b_HLT1EMHighEt;   //!
  TBranch        *b_HLT1EMVeryHighEt;   //!
  TBranch        *b_HLT2ElectronZCounter;   //!
  TBranch        *b_HLT2ElectronExclusive;   //!
  TBranch        *b_HLT2PhotonExclusive;   //!
  TBranch        *b_HLT1PhotonL1Isolated;   //!
  TBranch        *b_CandHLT1ElectronStartup;   //!
  TBranch        *b_CandHLT1ElectronRelaxedStartup;   //!
  TBranch        *b_CandHLT2ElectronStartup;   //!
  TBranch        *b_CandHLT2ElectronRelaxedStartup;   //!
  TBranch        *b_HLT1MuonIso;   //!
  TBranch        *b_HLT1MuonNonIso;   //!
  TBranch        *b_HLT2MuonIso;   //!
  TBranch        *b_HLT2MuonNonIso;   //!
  TBranch        *b_HLT2MuonJPsi;   //!
  TBranch        *b_HLT2MuonUpsilon;   //!
  TBranch        *b_HLT2MuonZ;   //!
  TBranch        *b_HLTNMuonNonIso;   //!
  TBranch        *b_HLT2MuonSameSign;   //!
  TBranch        *b_HLT1MuonPrescalePt3;   //!
  TBranch        *b_HLT1MuonPrescalePt5;   //!
  TBranch        *b_HLT1MuonPrescalePt7x7;   //!
  TBranch        *b_HLT1MuonPrescalePt7x10;   //!
  TBranch        *b_HLT1MuonLevel1;   //!
  TBranch        *b_CandHLT1MuonPrescaleVtx2cm;   //!
  TBranch        *b_CandHLT1MuonPrescaleVtx2mm;   //!
  TBranch        *b_CandHLT2MuonPrescaleVtx2cm;   //!
  TBranch        *b_CandHLT2MuonPrescaleVtx2mm;   //!
  TBranch        *b_HLTB1Jet;   //!
  TBranch        *b_HLTB2Jet;   //!
  TBranch        *b_HLTB3Jet;   //!
  TBranch        *b_HLTB4Jet;   //!
  TBranch        *b_HLTBHT;   //!
  TBranch        *b_HLTB1JetMu;   //!
  TBranch        *b_HLTB2JetMu;   //!
  TBranch        *b_HLTB3JetMu;   //!
  TBranch        *b_HLTB4JetMu;   //!
  TBranch        *b_HLTBHTMu;   //!
  TBranch        *b_HLTBJPsiMuMu;   //!
  TBranch        *b_HLT1Tau;   //!
  TBranch        *b_HLT1Tau1MET;   //!
  TBranch        *b_HLT2TauPixel;   //!
  TBranch        *b_HLTXElectronBJet;   //!
  TBranch        *b_HLTXMuonBJet;   //!
  TBranch        *b_HLTXMuonBJetSoftMuon;   //!
  TBranch        *b_HLTXElectron1Jet;   //!
  TBranch        *b_HLTXElectron2Jet;   //!
  TBranch        *b_HLTXElectron3Jet;   //!
  TBranch        *b_HLTXElectron4Jet;   //!
  TBranch        *b_HLTXMuonJets;   //!
  TBranch        *b_CandHLTXMuonNoL2IsoJets;   //!
  TBranch        *b_CandHLTXMuonNoIsoJets;   //!
  TBranch        *b_HLTXElectronMuon;   //!
  TBranch        *b_HLTXElectronMuonRelaxed;   //!
  TBranch        *b_HLTXElectronTau;   //!
  TBranch        *b_CandHLTXElectronTauPixel;   //!
  TBranch        *b_HLTXMuonTau;   //!
  TBranch        *b_CandHLTEcalPi0;   //!
  TBranch        *b_CandHLTEcalPhiSym;   //!
  TBranch        *b_CandHLTHcalPhiSym;   //!
  TBranch        *b_HLTHcalIsolatedTrack;   //!
  TBranch        *b_CandHLTHcalIsolatedTrackNoEcalIsol;   //!
  TBranch        *b_HLTMinBiasPixel;   //!
  TBranch        *b_CandHLTMinBiasForAlignment;   //!
  TBranch        *b_HLTMinBias;   //!
  TBranch        *b_HLTZeroBias;   //!
  TBranch        *b_HLTriggerType;   //!
  TBranch        *b_TriggerFinalPath;   //!
  //L1's
  TBranch        *b_L1_DoubleEG10_00001;   //! 
  TBranch        *b_L1_DoubleEG1_00001;   //! 
  TBranch        *b_L1_DoubleEG5_00001;   //! 
  TBranch        *b_L1_DoubleForJet20;   //! 
  TBranch        *b_L1_DoubleHfBitCountsRing1_P1N1;   //! 
  TBranch        *b_L1_DoubleHfBitCountsRing2_P1N1;   //! 
  TBranch        *b_L1_DoubleHfRingEtSumsRing1_P200N200;   //! 
  TBranch        *b_L1_DoubleHfRingEtSumsRing1_P4N4;   //! 
  TBranch        *b_L1_DoubleHfRingEtSumsRing2_P200N200;   //! 
  TBranch        *b_L1_DoubleHfRingEtSumsRing2_P4N4;   //! 
  TBranch        *b_L1_DoubleIsoEG05_TopBottom;   //! 
  TBranch        *b_L1_DoubleIsoEG05_TopBottomCen;   //! 
  TBranch        *b_L1_DoubleIsoEG10_00001;   //! 
  TBranch        *b_L1_DoubleIsoEG8_00001;   //! 
  TBranch        *b_L1_DoubleJet40_00001;   //! 
  TBranch        *b_L1_DoubleJet60_00001;   //! 
  TBranch        *b_L1_DoubleMu3;   //! 
  TBranch        *b_L1_DoubleMuOpen;   //! 
  TBranch        *b_L1_DoubleMuTopBottom;   //! 
  TBranch        *b_L1_DoubleNoIsoEG05_TopBottom;   //! 
  TBranch        *b_L1_DoubleNoIsoEG05_TopBottomCen;   //! 
  TBranch        *b_L1_DoubleTauJet20_00001;   //! 
  TBranch        *b_L1_DoubleTauJet8_00001;   //! 
  TBranch        *b_L1_EG12_Jet40_00001;   //! 
  TBranch        *b_L1_EG5_TripleJet6_00001;   //! 
  TBranch        *b_L1_ETM20_00001;   //! 
  TBranch        *b_L1_ETM30_00001;   //! 
  TBranch        *b_L1_ETM40_00001;   //! 
  TBranch        *b_L1_ETM50_00001;   //! 
  TBranch        *b_L1_ETT60_00001;   //! 
  TBranch        *b_L1_HTT100_00001;   //! 
  TBranch        *b_L1_HTT200_00001;   //! 
  TBranch        *b_L1_HTT300_00001;   //! 
  TBranch        *b_L1_IsoEG10_Jet12_00001;   //! 
  TBranch        *b_L1_IsoEG10_Jet6_00001;   //! 
  TBranch        *b_L1_IsoEG10_Jet6_ForJet6_00001;   //! 
  TBranch        *b_L1_IsoEG10_Jet8_00001;   //! 
  TBranch        *b_L1_IsoEG10_TauJet8_00001;   //! 
  TBranch        *b_L1_MinBias_ETT10_00001;   //! 
  TBranch        *b_L1_MinBias_HTT10_00001;   //! 
  TBranch        *b_L1_Mu3_EG12_00001;   //! 
  TBranch        *b_L1_Mu3_IsoEG5_00001;   //! 
  TBranch        *b_L1_Mu3_Jet6_00001;   //! 
  TBranch        *b_L1_Mu3_TripleJet6_00001;   //! 
  TBranch        *b_L1_Mu5_IsoEG10_00001;   //! 
  TBranch        *b_L1_Mu5_Jet6_00001;   //! 
  TBranch        *b_L1_Mu5_TauJet8_00001;   //! 
  TBranch        *b_L1_QuadJet20_00001;   //! 
  TBranch        *b_L1_QuadJet6_00001;   //! 
  TBranch        *b_L1_SingleEG1;   //! 
  TBranch        *b_L1_SingleEG10_00001;   //! 
  TBranch        *b_L1_SingleEG12_00001;   //! 
  TBranch        *b_L1_SingleEG15_00001;   //! 
  TBranch        *b_L1_SingleEG20_00001;   //! 
  TBranch        *b_L1_SingleEG5_00001;   //! 
  TBranch        *b_L1_SingleEG5_Endcap_00001;   //! 
  TBranch        *b_L1_SingleEG8_00001;   //! 
  TBranch        *b_L1_SingleForJet10;   //! 
  TBranch        *b_L1_SingleForJet6;   //! 
  TBranch        *b_L1_SingleHfBitCountsRing1_1;   //! 
  TBranch        *b_L1_SingleHfBitCountsRing2_1;   //! 
  TBranch        *b_L1_SingleHfRingEtSumsRing1_200;   //! 
  TBranch        *b_L1_SingleHfRingEtSumsRing1_4;   //! 
  TBranch        *b_L1_SingleHfRingEtSumsRing2_200;   //! 
  TBranch        *b_L1_SingleHfRingEtSumsRing2_4;   //! 
  TBranch        *b_L1_SingleIsoEG10_00001;   //! 
  TBranch        *b_L1_SingleIsoEG12_00001;   //! 
  TBranch        *b_L1_SingleIsoEG15_00001;   //! 
  TBranch        *b_L1_SingleIsoEG5_00001;   //! 
  TBranch        *b_L1_SingleIsoEG5_Endcap_00001;   //! 
  TBranch        *b_L1_SingleIsoEG8_00001;   //! 
  TBranch        *b_L1_SingleJet10_00001;   //! 
  TBranch        *b_L1_SingleJet10_Barrel_00001;   //! 
  TBranch        *b_L1_SingleJet10_Central;   //! 
  TBranch        *b_L1_SingleJet10_Endcap;   //! 
  TBranch        *b_L1_SingleJet20_00001;   //! 
  TBranch        *b_L1_SingleJet20_Barrel_00001;   //! 
  TBranch        *b_L1_SingleJet30_00001;   //! 
  TBranch        *b_L1_SingleJet30_Barrel_00001;   //! 
  TBranch        *b_L1_SingleJet40_00001;   //! 
  TBranch        *b_L1_SingleJet40_Barrel_00001;   //! 
  TBranch        *b_L1_SingleJet50_00001;   //! 
  TBranch        *b_L1_SingleJet60_00001;   //! 
  TBranch        *b_L1_SingleJet6_00001;   //! 
  TBranch        *b_L1_SingleJet6_Barrel_00001;   //! 
  TBranch        *b_L1_SingleJet6_Central;   //! 
  TBranch        *b_L1_SingleJet6_Endcap;   //! 
  TBranch        *b_L1_SingleMu0;   //! 
  TBranch        *b_L1_SingleMu10;   //! 
  TBranch        *b_L1_SingleMu14;   //! 
  TBranch        *b_L1_SingleMu3;   //! 
  TBranch        *b_L1_SingleMu5;   //! 
  TBranch        *b_L1_SingleMu7;   //! 
  TBranch        *b_L1_SingleMuBeamHalo;   //! 
  TBranch        *b_L1_SingleMuOpen;   //! 
  TBranch        *b_L1_SingleTauJet10_00001;   //! 
  TBranch        *b_L1_SingleTauJet10_Barrel_00001;   //! 
  TBranch        *b_L1_SingleTauJet20_00001;   //! 
  TBranch        *b_L1_SingleTauJet20_Barrel_00001;   //! 
  TBranch        *b_L1_SingleTauJet30_00001;   //! 
  TBranch        *b_L1_SingleTauJet30_Barrel_00001;   //! 
  TBranch        *b_L1_SingleTauJet50_00001;   //! 
  TBranch        *b_L1_SingleTauJet8_00001;   //! 
  TBranch        *b_L1_SingleTauJet8_Barrel_00001;   //! 
  TBranch        *b_L1_TauJet10_ETM30_00001;   //! 
  TBranch        *b_L1_TauJet10_ETM40_00001;   //! 
  TBranch        *b_L1_TripleJet30_00001;   //! 
  TBranch        *b_L1_TripleMu3;   //! 

  // 21X HLT names
  TBranch        *b_HLT_L1Jet15;   //!
  TBranch        *b_HLT_Jet30;   //!
  TBranch        *b_HLT_Jet50;   //!
  TBranch        *b_HLT_Jet80;   //!
  TBranch        *b_HLT_Jet110;   //!
  TBranch        *b_HLT_Jet180;   //!
  TBranch        *b_HLT_Jet250;   //!
  TBranch        *b_HLT_FwdJet20;   //!
  TBranch        *b_HLT_DoubleJet150;   //!
  TBranch        *b_HLT_DoubleJet125_Aco;   //!
  TBranch        *b_HLT_DoubleFwdJet50;   //!
  TBranch        *b_HLT_DiJetAve15;   //!
  TBranch        *b_HLT_DiJetAve30;   //!
  TBranch        *b_HLT_DiJetAve50;   //!
  TBranch        *b_HLT_DiJetAve70;   //!
  TBranch        *b_HLT_DiJetAve130;   //!
  TBranch        *b_HLT_DiJetAve220;   //!
  TBranch        *b_HLT_TripleJet85;   //!
  TBranch        *b_HLT_QuadJet30;   //!
  TBranch        *b_HLT_QuadJet60;   //!
  TBranch        *b_HLT_SumET120;   //!
  TBranch        *b_HLT_L1MET20;   //!
  TBranch        *b_HLT_MET25;   //!
  TBranch        *b_HLT_MET35;   //!
  TBranch        *b_HLT_MET50;   //!
  TBranch        *b_HLT_MET65;   //!
  TBranch        *b_HLT_MET75;   //!
  TBranch        *b_HLT_MET35_HT350;   //!
  TBranch        *b_HLT_Jet180_MET60;   //!
  TBranch        *b_HLT_Jet60_MET70_Aco;   //!
  TBranch        *b_HLT_Jet100_MET60_Aco;   //!
  TBranch        *b_HLT_DoubleJet125_MET60;   //!
  TBranch        *b_HLT_DoubleFwdJet40_MET60;   //!
  TBranch        *b_HLT_DoubleJet60_MET60_Aco;   //!
  TBranch        *b_HLT_DoubleJet50_MET70_Aco;   //!
  TBranch        *b_HLT_DoubleJet40_MET70_Aco;   //!
  TBranch        *b_HLT_TripleJet60_MET60;   //!
  TBranch        *b_HLT_QuadJet35_MET60;   //!
  TBranch        *b_HLT_IsoEle15_L1I;   //!
  TBranch        *b_HLT_IsoEle18_L1R;   //!
  TBranch        *b_HLT_IsoEle15_LW_L1I;   //!
  TBranch        *b_HLT_LooseIsoEle15_LW_L1R;   //!
  TBranch        *b_HLT_Ele10_SW_L1R;   //!
  TBranch        *b_HLT_Ele15_SW_L1R;   //!
  TBranch        *b_HLT_Ele15_LW_L1R;   //!
  TBranch        *b_HLT_EM80;   //!
  TBranch        *b_HLT_EM200;   //!
  TBranch        *b_HLT_DoubleIsoEle10_L1I;   //!
  TBranch        *b_HLT_DoubleIsoEle12_L1R;   //!
  TBranch        *b_HLT_DoubleIsoEle10_LW_L1I;   //!
  TBranch        *b_HLT_DoubleIsoEle12_LW_L1R;   //!
  TBranch        *b_HLT_DoubleEle5_SW_L1R;   //!
  TBranch        *b_HLT_DoubleEle10_LW_OnlyPixelM_L1R;   //!
  TBranch        *b_HLT_DoubleEle10_Z;   //!
  TBranch        *b_HLT_DoubleEle6_Exclusive;   //!
  TBranch        *b_HLT_IsoPhoton30_L1I;   //!
  TBranch        *b_HLT_IsoPhoton10_L1R;   //!
  TBranch        *b_HLT_IsoPhoton15_L1R;   //!
  TBranch        *b_HLT_IsoPhoton20_L1R;   //!
  TBranch        *b_HLT_IsoPhoton25_L1R;   //!
  TBranch        *b_HLT_IsoPhoton40_L1R;   //!
  TBranch        *b_HLT_Photon15_L1R;   //!
  TBranch        *b_HLT_Photon25_L1R;   //!
  TBranch        *b_HLT_DoubleIsoPhoton20_L1I;   //!
  TBranch        *b_HLT_DoubleIsoPhoton20_L1R;   //!
  TBranch        *b_HLT_DoublePhoton10_Exclusive;   //!
  TBranch        *b_HLT_L1Mu;   //!
  TBranch        *b_HLT_L1MuOpen;   //!
  TBranch        *b_HLT_L2Mu9;   //!
  TBranch        *b_HLT_IsoMu9;   //!
  TBranch        *b_HLT_IsoMu11;   //!
  TBranch        *b_HLT_IsoMu13;   //!
  TBranch        *b_HLT_IsoMu15;   //!
  TBranch        *b_HLT_NoTrackerIsoMu15;   //!
  TBranch        *b_HLT_Mu3;   //!
  TBranch        *b_HLT_Mu5;   //!
  TBranch        *b_HLT_Mu7;   //!
  TBranch        *b_HLT_Mu9;   //!
  TBranch        *b_HLT_Mu11;   //!
  TBranch        *b_HLT_Mu13;   //!
  TBranch        *b_HLT_Mu15;   //!
  TBranch        *b_HLT_Mu15_L1Mu7;   //!
  TBranch        *b_HLT_Mu15_Vtx2cm;   //!
  TBranch        *b_HLT_Mu15_Vtx2mm;   //!
  TBranch        *b_HLT_DoubleIsoMu3;   //!
  TBranch        *b_HLT_DoubleMu3;   //!
  TBranch        *b_HLT_DoubleMu3_Vtx2cm;   //!
  TBranch        *b_HLT_DoubleMu3_Vtx2mm;   //!
  TBranch        *b_HLT_DoubleMu3_JPsi;   //!
  TBranch        *b_HLT_DoubleMu3_Upsilon;   //!
  TBranch        *b_HLT_DoubleMu7_Z;   //!
  TBranch        *b_HLT_DoubleMu3_SameSign;   //!
  TBranch        *b_HLT_DoubleMu3_Psi2S;   //!
  TBranch        *b_HLT_BTagIP_Jet180;   //!
  TBranch        *b_HLT_BTagIP_Jet120_Relaxed;   //!
  TBranch        *b_HLT_BTagIP_DoubleJet120;   //!
  TBranch        *b_HLT_BTagIP_DoubleJet60_Relaxed;   //!
  TBranch        *b_HLT_BTagIP_TripleJet70;   //!
  TBranch        *b_HLT_BTagIP_TripleJet40_Relaxed;   //!
  TBranch        *b_HLT_BTagIP_QuadJet40;   //!
  TBranch        *b_HLT_BTagIP_QuadJet30_Relaxed;   //!
  TBranch        *b_HLT_BTagIP_HT470;   //!
  TBranch        *b_HLT_BTagIP_HT320_Relaxed;   //!
  TBranch        *b_HLT_BTagMu_DoubleJet120;   //!
  TBranch        *b_HLT_BTagMu_DoubleJet60_Relaxed;   //!
  TBranch        *b_HLT_BTagMu_TripleJet70;   //!
  TBranch        *b_HLT_BTagMu_TripleJet40_Relaxed;   //!
  TBranch        *b_HLT_BTagMu_QuadJet40;   //!
  TBranch        *b_HLT_BTagMu_QuadJet30_Relaxed;   //!
  TBranch        *b_HLT_BTagMu_HT370;   //!
  TBranch        *b_HLT_BTagMu_HT250_Relaxed;   //!
  TBranch        *b_HLT_DoubleMu3_BJPsi;   //!
  TBranch        *b_HLT_DoubleMu4_BJPsi;   //!
  TBranch        *b_HLT_TripleMu3_TauTo3Mu;   //!
  TBranch        *b_HLT_IsoTau_MET65_Trk20;   //!
  TBranch        *b_HLT_IsoTau_MET35_Trk15_L1MET;   //!
  TBranch        *b_HLT_LooseIsoTau_MET30;   //!
  TBranch        *b_HLT_LooseIsoTau_MET30_L1MET;   //!
  TBranch        *b_HLT_DoubleIsoTau_Trk3;   //!
  TBranch        *b_HLT_DoubleLooseIsoTau;   //!
  TBranch        *b_HLT_IsoEle8_IsoMu7;   //!
  TBranch        *b_HLT_IsoEle10_Mu10_L1R;   //!
  TBranch        *b_HLT_IsoEle12_IsoTau_Trk3;   //!
  TBranch        *b_HLT_IsoEle10_BTagIP_Jet35;   //!
  TBranch        *b_HLT_IsoEle12_Jet40;   //!
  TBranch        *b_HLT_IsoEle12_DoubleJet80;   //!
  TBranch        *b_HLT_IsoElec5_TripleJet30;   //!
  TBranch        *b_HLT_IsoEle12_TripleJet60;   //!
  TBranch        *b_HLT_IsoEle12_QuadJet35;   //!
  TBranch        *b_HLT_IsoMu14_IsoTau_Trk3;   //!
  TBranch        *b_HLT_IsoMu7_BTagIP_Jet35;   //!
  TBranch        *b_HLT_IsoMu7_BTagMu_Jet20;   //!
  TBranch        *b_HLT_IsoMu7_Jet40;   //!
  TBranch        *b_HLT_NoL2IsoMu8_Jet40;   //!
  TBranch        *b_HLT_Mu14_Jet50;   //!
  TBranch        *b_HLT_Mu5_TripleJet30;   //!
  TBranch        *b_HLT_BTagMu_Jet20_Calib;   //!
  TBranch        *b_HLT_ZeroBias;   //!
  TBranch        *b_HLT_MinBias;   //!
  TBranch        *b_HLT_MinBiasHcal;   //!
  TBranch        *b_HLT_MinBiasEcal;   //!
  TBranch        *b_HLT_MinBiasPixel;   //!
  TBranch        *b_HLT_MinBiasPixel_Trk5;   //!
  TBranch        *b_HLT_BackwardBSC;   //!
  TBranch        *b_HLT_ForwardBSC;   //!
  TBranch        *b_HLT_CSCBeamHalo;   //!
  TBranch        *b_HLT_CSCBeamHaloOverlapRing1;   //!
  TBranch        *b_HLT_CSCBeamHaloOverlapRing2;   //!
  TBranch        *b_HLT_CSCBeamHaloRing2or3;   //!
  TBranch        *b_HLT_TrackerCosmics;   //!
  TBranch        *b_HLT_TriggerType;   //!
  TBranch        *b_AlCa_IsoTrack;   //!
  TBranch        *b_AlCa_EcalPhiSym;   //!
  TBranch        *b_AlCa_EcalPi0;   //!

  // Cut on mu quality
  Int_t           NL1OpenMu;
  Float_t         L1OpenMuPt[10];   //[NL1OpenMu]
  Float_t         L1OpenMuE[10];   //[NL1OpenMu]
  Float_t         L1OpenMuEta[10];   //[NL1OpenMu]
  Float_t         L1OpenMuPhi[10];   //[NL1OpenMu]
  Int_t           L1OpenMuIsol[10];   //[NL1OpenMu]
  Int_t           L1OpenMuMip[10];   //[NL1OpenMu]
  Int_t           L1OpenMuFor[10];   //[NL1OpenMu]
  Int_t           L1OpenMuRPC[10];   //[NL1OpenMu]
  Int_t           L1OpenMuQal[10];   //[NL1OpenMu]

  Int_t           NL1GoodSingleMu;
  Float_t         L1GoodSingleMuPt[10];   //[NL1GoodSingleMu]
  Float_t         L1GoodSingleMuE[10];   //[NL1GoodSingleMu]
  Float_t         L1GoodSingleMuEta[10];   //[NL1GoodSingleMu]
  Float_t         L1GoodSingleMuPhi[10];   //[NL1GoodSingleMu]
  Int_t           L1GoodSingleMuIsol[10];   //[NL1GoodSingleMu]
  Int_t           L1GoodSingleMuMip[10];   //[NL1GoodSingleMu]
  Int_t           L1GoodSingleMuFor[10];   //[NL1GoodSingleMu]
  Int_t           L1GoodSingleMuRPC[10];   //[NL1GoodSingleMu]
  Int_t           L1GoodSingleMuQal[10];   //[NL1GoodSingleMu]

  Int_t           NL1GoodDoubleMu;
  Float_t         L1GoodDoubleMuPt[10];   //[NL1GoodDoubleMu]
  Float_t         L1GoodDoubleMuE[10];   //[NL1GoodDoubleMu]
  Float_t         L1GoodDoubleMuEta[10];   //[NL1GoodDoubleMu]
  Float_t         L1GoodDoubleMuPhi[10];   //[NL1GoodDoubleMu]
  Int_t           L1GoodDoubleMuIsol[10];   //[NL1GoodDoubleMu]
  Int_t           L1GoodDoubleMuMip[10];   //[NL1GoodDoubleMu]
  Int_t           L1GoodDoubleMuFor[10];   //[NL1GoodDoubleMu]
  Int_t           L1GoodDoubleMuRPC[10];   //[NL1GoodDoubleMu]
  Int_t           L1GoodDoubleMuQal[10];   //[NL1GoodDoubleMu]



  OHltTree(TTree *tree=0, OHltMenu *menu=0);
  virtual ~OHltTree();
  virtual Bool_t   Notify();
  virtual void     Init(TTree *tree);

  inline Int_t     GetEntry(Long64_t entry);
  inline Long64_t  LoadTree(Long64_t entry);
  inline void	   SetMapL1BitOfStandardHLTPath(OHltMenu *menu);
  inline void	   SetMapL1SeedsOfStandardHLTPath(OHltMenu *menu);
  inline void ApplyL1Prescales(OHltMenu *menu);
  inline void SetL1MuonQuality();
  inline void SetOpenL1Bits();

  void Loop(OHltRateCounter *rc,OHltConfig *cfg,OHltMenu *menu,int pID);

  void CheckOpenHlt(OHltConfig *cfg,OHltMenu *menu,int it);
  void PrintOhltVariables(int level, int type);
  int OpenHltTauPassed(float Et,float Eiso, float L25Tpt, int L25Tiso,float L3Tpt, int L3Tiso);
  int OHltTree::OpenHltTauL2SCPassed(float Et,float L25Tpt, int L25Tiso, float L3Tpt, int L3Tiso);
  int OHltTree::OpenHltElecTauL2SCPassed(float elecEt, int elecL1iso, float elecTiso, float elecHiso,
					 float tauEt,float tauL25Tpt, int tauL25Tiso, float tauL3Tpt, int tauL3Tiso);
  int OpenHlt1ElectronPassed(float Et,int L1iso,float Tiso,float Hiso);
  int OpenHlt1LWElectronPassed(float Et,int L1iso,float Tiso,float Hiso); 
  int OpenHlt1PhotonPassed(float Et,int L1iso,float Tiso,float Eiso,float HisoBR,float HisoEC);
  int OpenHlt1PhotonLooseEcalIsoPassed(float Et,int L1iso,float Tiso,float Eiso,float HisoBR,float HisoEC);
  int OpenHlt1MuonPassed(double ptl1,double ptl2,double ptl3,double dr,int iso);
  int OpenHlt2MuonPassed(double ptl1,double ptl2,double ptl3,double dr,int iso);
  int OpenHlt1L2MuonPassed(double ptl1,double ptl2,double dr);  
  int OpenHlt1JetPassed(double pt);
  int OpenHlt1CorJetPassed(double pt);
  int OpenHltFwdJetPassed(double esum);
  int OpenHltDiJetAvePassed(double pt);
  int OpenHltCorDiJetAvePassed(double pt);
  int OpenHltQuadJetPassed(double pt);
  int OpenHltJRMuonPassed(double ptl1,double ptl2,double ptl3,double dr,int iso,double ptl3hi);

  std::map<TString, std::vector<TString> >&
    GetL1SeedsOfHLTPathMap() { return map_L1SeedsOfStandardHLTPath; }; // mapping to all seeds


private:

  int nTrig;
  int nL1Trig;
  std::vector<int> triggerBit;
  std::vector<int> previousBitsFired;
  std::vector<int> allOtherBitsFired;
  std::vector<int> BitOfStandardHLTPath;
  std::map<TString,int> map_BitOfStandardHLTPath;
  std::map<TString,int> map_L1BitOfStandardHLTPath;

  std::map<TString, std::vector<TString> > map_L1SeedsOfStandardHLTPath; // mapping to all seeds

  TRandom3 random; // for random prescale method
  inline int GetIntRandom() { return (int)(9999999.*random.Rndm()); }

  enum e_objType {muon,electron,tau,photon,jet};

};

#ifdef OHltTree_cxx
OHltTree::OHltTree(TTree *tree, OHltMenu *menu)
{
  cout<<"Initialising OHltTree."<<endl;
  if (tree == 0) {
    cerr<<"Error initialising tree!"<<endl;
    return;
  }
  if (menu == 0) {
    cerr<<"Error: no menu!"<<endl;
    return;
  }
  Init(tree);
  
  nTrig = menu->GetTriggerSize();
  nL1Trig = menu->GetL1TriggerSize();

  triggerBit.reserve(nTrig);
  previousBitsFired.reserve(nTrig);
  allOtherBitsFired.reserve(nTrig);
  BitOfStandardHLTPath.reserve(nTrig);
  
  for (int it = 0; it < nTrig; it++){
    triggerBit.push_back(false);
    previousBitsFired.push_back(false);
    allOtherBitsFired.push_back(false);
  }  

  SetMapL1SeedsOfStandardHLTPath(menu);

  cout<<"Succeeded initialising OHltTree. nEntries: "<<fChain->GetEntries()<<endl;
}

OHltTree::~OHltTree()
{
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t OHltTree::GetEntry(Long64_t entry)
{
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t OHltTree::LoadTree(Long64_t entry)
{
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (!fChain->InheritsFrom(TChain::Class()))  return centry;
  TChain *chain = (TChain*)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void OHltTree::Init(TTree *tree)
{
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("NrecoJetCal", &NrecoJetCal, &b_NrecoJetCal);
  fChain->SetBranchAddress("NrecoJetGen", &NrecoJetGen, &b_NrecoJetGen);
  fChain->SetBranchAddress("NrecoTowCal", &NrecoTowCal, &b_NrecoTowCal);
  fChain->SetBranchAddress("NrecoJetCorCal", &NrecoJetCorCal, &b_NrecoJetCorCal);
  fChain->SetBranchAddress("recoJetCalPt", recoJetCalPt, &b_recoJetCalPt);
  fChain->SetBranchAddress("recoJetCalPhi", recoJetCalPhi, &b_recoJetCalPhi);
  fChain->SetBranchAddress("recoJetCalEta", recoJetCalEta, &b_recoJetCalEta);
  fChain->SetBranchAddress("recoJetCalEt", recoJetCalEt, &b_recoJetCalEt);
  fChain->SetBranchAddress("recoJetCalE", recoJetCalE, &b_recoJetCalE);
  fChain->SetBranchAddress("recoJetGenPt", &recoJetGenPt, &b_recoJetGenPt);
  fChain->SetBranchAddress("recoJetGenPhi", &recoJetGenPhi, &b_recoJetGenPhi);
  fChain->SetBranchAddress("recoJetGenEta", &recoJetGenEta, &b_recoJetGenEta);
  fChain->SetBranchAddress("recoJetGenEt", &recoJetGenEt, &b_recoJetGenEt);
  fChain->SetBranchAddress("recoJetGenE", &recoJetGenE, &b_recoJetGenE);
  fChain->SetBranchAddress("recoTowEt", &recoTowEt, &b_recoTowEt);
  fChain->SetBranchAddress("recoTowEta", &recoTowEta, &b_recoTowEta);
  fChain->SetBranchAddress("recoTowPhi", &recoTowPhi, &b_recoTowPhi);
  fChain->SetBranchAddress("recoTowE", &recoTowE, &b_recoTowE);
  fChain->SetBranchAddress("recoTowEm", &recoTowEm, &b_recoTowEm);
  fChain->SetBranchAddress("recoTowHad", &recoTowHad, &b_recoTowHad);
  fChain->SetBranchAddress("recoTowOE", &recoTowOE, &b_recoTowOE);
  fChain->SetBranchAddress("recoJetCorCalPt", recoJetCorCalPt, &b_recoJetCorCalPt);
  fChain->SetBranchAddress("recoJetCorCalPhi", recoJetCorCalPhi, &b_recoJetCorCalPhi);
  fChain->SetBranchAddress("recoJetCorCalEta", recoJetCorCalEta, &b_recoJetCorCalEta);
  fChain->SetBranchAddress("recoJetCorCalE", recoJetCorCalE, &b_recoJetCorCalE);
  fChain->SetBranchAddress("recoMetCal", &recoMetCal, &b_recoMetCal);
  fChain->SetBranchAddress("recoMetCalPhi", &recoMetCalPhi, &b_recoMetCalPhi);
  fChain->SetBranchAddress("recoMetCalSum", &recoMetCalSum, &b_recoMetCalSum);
  fChain->SetBranchAddress("recoMetGen", &recoMetGen, &b_recoMetGen);
  fChain->SetBranchAddress("recoMetGenPhi", &recoMetGenPhi, &b_recoMetGenPhi);
  fChain->SetBranchAddress("recoMetGenSum", &recoMetGenSum, &b_recoMetGenSum);
  fChain->SetBranchAddress("recoHTCal", &recoHTCal, &b_recoHTCal);
  fChain->SetBranchAddress("recoHTCalPhi", &recoHTCalPhi, &b_recoHTCalPhi);
  fChain->SetBranchAddress("recoHTCalSum", &recoHTCalSum, &b_recoHTCalSum);
  fChain->SetBranchAddress("NohTau", &NohTau, &b_NohTau);
  fChain->SetBranchAddress("ohTauEta", ohTauEta, &b_ohTauEta);
  fChain->SetBranchAddress("ohTauPhi", ohTauPhi, &b_ohTauPhi);
  fChain->SetBranchAddress("ohTauPt", ohTauPt, &b_ohTauPt);
  fChain->SetBranchAddress("ohTauEiso", ohTauEiso, &b_ohTauEiso);
  fChain->SetBranchAddress("ohTauL25Tpt", ohTauL25Tpt, &b_ohTauL25Tpt);
  fChain->SetBranchAddress("ohTauL25Tiso", ohTauL25Tiso, &b_ohTauL25Tiso);
  fChain->SetBranchAddress("ohTauL3Tpt", ohTauL3Tpt, &b_ohTauL3Tpt);
  fChain->SetBranchAddress("ohTauL3Tiso", ohTauL3Tiso, &b_ohTauL3Tiso);
  fChain->SetBranchAddress("NohBJetL2", &NohBJetL2, &b_NohBJetL2);
  fChain->SetBranchAddress("ohBJetL2CorrectedEt", &ohBJetL2CorrectedEt, &b_ohBJetL2CorrectedEt);
  fChain->SetBranchAddress("ohBJetL2Et", &ohBJetL2Et, &b_ohBJetL2Et);
  fChain->SetBranchAddress("NohBJetLife", &NohBJetLife, &b_NohBJetLife);
  fChain->SetBranchAddress("ohBJetLifeL2E", ohBJetLifeL2E, &b_ohBJetLifeL2E);
  fChain->SetBranchAddress("ohBJetLifeL2ET", ohBJetLifeL2ET, &b_ohBJetLifeL2ET);
  fChain->SetBranchAddress("ohBJetLifeL2Eta", ohBJetLifeL2Eta, &b_ohBJetLifeL2Eta);
  fChain->SetBranchAddress("ohBJetLifeL2Phi", ohBJetLifeL2Phi, &b_ohBJetLifeL2Phi);
  fChain->SetBranchAddress("ohBJetLifeL25Discriminator", ohBJetLifeL25Discriminator, &b_ohBJetLifeL25Discriminator);
  fChain->SetBranchAddress("ohBJetLifeL3Discriminator", ohBJetLifeL3Discriminator, &b_ohBJetLifeL3Discriminator);
  fChain->SetBranchAddress("NohBJetPixelTracks", &NohBJetPixelTracks, &b_NohBJetPixelTracks);
  fChain->SetBranchAddress("ohBJetLifePixelTrackPt", ohBJetLifePixelTrackPt, &b_ohBJetLifePixelTrackPt);
  fChain->SetBranchAddress("ohBJetLifePixelTrackEta", ohBJetLifePixelTrackEta, &b_ohBJetLifePixelTrackEta);
  fChain->SetBranchAddress("ohBJetLifePixelTrackPhi", ohBJetLifePixelTrackPhi, &b_ohBJetLifePixelTrackPhi);
  fChain->SetBranchAddress("ohBJetLifePixelTrackChi2", ohBJetLifePixelTrackChi2, &b_ohBJetLifePixelTrackChi2);
  fChain->SetBranchAddress("NohBJetRegionalTracks", &NohBJetRegionalTracks, &b_NohBJetRegionalTracks);
  fChain->SetBranchAddress("ohBJetLifeRegionalTrackPt", ohBJetLifeRegionalTrackPt, &b_ohBJetLifeRegionalTrackPt);
  fChain->SetBranchAddress("ohBJetLifeRegionalTrackEta", ohBJetLifeRegionalTrackEta, &b_ohBJetLifeRegionalTrackEta);
  fChain->SetBranchAddress("ohBJetLifeRegionalTrackPhi", ohBJetLifeRegionalTrackPhi, &b_ohBJetLifeRegionalTrackPhi);
  fChain->SetBranchAddress("ohBJetLifeRegionalTrackChi2", ohBJetLifeRegionalTrackChi2, &b_ohBJetLifeRegionalTrackChi2);
  fChain->SetBranchAddress("ohBJetLifeRegionalSeedPt", ohBJetLifeRegionalSeedPt, &b_ohBJetLifeRegionalSeedPt);
  fChain->SetBranchAddress("ohBJetLifeRegionalSeedEta", ohBJetLifeRegionalSeedEta, &b_ohBJetLifeRegionalSeedEta);
  fChain->SetBranchAddress("ohBJetLifeRegionalSeedPhi", ohBJetLifeRegionalSeedPhi, &b_ohBJetLifeRegionalSeedPhi);
  fChain->SetBranchAddress("NohBJetSoftm", &NohBJetSoftm, &b_NohBJetSoftm);
  fChain->SetBranchAddress("ohBJetSoftmL2E", ohBJetSoftmL2E, &b_ohBJetSoftmL2E);
  fChain->SetBranchAddress("ohBJetSoftmL2ET", ohBJetSoftmL2ET, &b_ohBJetSoftmL2ET);
  fChain->SetBranchAddress("ohBJetSoftmL2Eta", ohBJetSoftmL2Eta, &b_ohBJetSoftmL2Eta);
  fChain->SetBranchAddress("ohBJetSoftmL2Phi", ohBJetSoftmL2Phi, &b_ohBJetSoftmL2Phi);
  fChain->SetBranchAddress("ohBJetSoftmL25Discriminator", ohBJetSoftmL25Discriminator, &b_ohBJetSoftmL25Discriminator);
  fChain->SetBranchAddress("ohBJetSoftmL3Discriminator", ohBJetSoftmL3Discriminator, &b_ohBJetSoftmL3Discriminator);
  fChain->SetBranchAddress("NohBJetL2Corrected", &NohBJetL2Corrected, &b_NohBJetL2Corrected);
  fChain->SetBranchAddress("ohBJetPerfL2E", ohBJetPerfL2E, &b_ohBJetPerfL2E);
  fChain->SetBranchAddress("ohBJetPerfL2ET", ohBJetPerfL2ET, &b_ohBJetPerfL2ET);
  fChain->SetBranchAddress("ohBJetPerfL2Eta", ohBJetPerfL2Eta, &b_ohBJetPerfL2Eta);
  fChain->SetBranchAddress("ohBJetPerfL2Phi", ohBJetPerfL2Phi, &b_ohBJetPerfL2Phi);
  fChain->SetBranchAddress("ohBJetPerfL25Tag", ohBJetPerfL25Tag, &b_ohBJetPerfL25Tag);
  fChain->SetBranchAddress("ohBJetPerfL3Tag", ohBJetPerfL3Tag, &b_ohBJetPerfL3Tag);
  fChain->SetBranchAddress("NrecoElec", &NrecoElec, &b_NrecoElec);
  fChain->SetBranchAddress("recoElecPt", &recoElecPt, &b_recoElecPt);
  fChain->SetBranchAddress("recoElecPhi", &recoElecPhi, &b_recoElecPhi);
  fChain->SetBranchAddress("recoElecEta", &recoElecEta, &b_recoElecEta);
  fChain->SetBranchAddress("recoElecEt", &recoElecEt, &b_recoElecEt);
  fChain->SetBranchAddress("recoElecE", &recoElecE, &b_recoElecE);
  fChain->SetBranchAddress("NrecoPhot", &NrecoPhot, &b_NrecoPhot);
  fChain->SetBranchAddress("recoPhotPt", &recoPhotPt, &b_recoPhotPt);
  fChain->SetBranchAddress("recoPhotPhi", &recoPhotPhi, &b_recoPhotPhi);
  fChain->SetBranchAddress("recoPhotEta", &recoPhotEta, &b_recoPhotEta);
  fChain->SetBranchAddress("recoPhotEt", &recoPhotEt, &b_recoPhotEt);
  fChain->SetBranchAddress("recoPhotE", &recoPhotE, &b_recoPhotE);
  fChain->SetBranchAddress("NohPhot", &NohPhot, &b_NohPhot);
  fChain->SetBranchAddress("ohPhotEt", ohPhotEt, &b_ohPhotEt);
  fChain->SetBranchAddress("ohPhotEta", ohPhotEta, &b_ohPhotEta);
  fChain->SetBranchAddress("ohPhotPhi", ohPhotPhi, &b_ohPhotPhi);
  fChain->SetBranchAddress("ohPhotEiso", ohPhotEiso, &b_ohPhotEiso);
  fChain->SetBranchAddress("ohPhotHiso", ohPhotHiso, &b_ohPhotHiso);
  fChain->SetBranchAddress("ohPhotTiso", ohPhotTiso, &b_ohPhotTiso);
  fChain->SetBranchAddress("ohPhotL1iso", ohPhotL1iso, &b_ohPhotL1iso);
  fChain->SetBranchAddress("NohEle", &NohEle, &b_NohEle);
  fChain->SetBranchAddress("ohEleEt", ohEleEt, &b_ohEleEt);
  fChain->SetBranchAddress("ohEleEta", ohEleEta, &b_ohEleEta);
  fChain->SetBranchAddress("ohElePhi", ohElePhi, &b_ohElePhi);
  fChain->SetBranchAddress("ohEleE", ohEleE, &b_ohEleE);
  fChain->SetBranchAddress("ohEleP", ohEleP, &b_ohEleP);
  fChain->SetBranchAddress("ohEleHiso", ohEleHiso, &b_ohEleHiso);
  fChain->SetBranchAddress("ohEleTiso", ohEleTiso, &b_ohEleTiso);
  fChain->SetBranchAddress("ohEleL1iso", ohEleL1iso, &b_ohEleL1iso);
  fChain->SetBranchAddress("ohElePixelSeeds", ohElePixelSeeds, &b_ohElePixelSeeds);
  fChain->SetBranchAddress("ohEleNewSC", ohEleNewSC, &b_ohEleNewSC);
  fChain->SetBranchAddress("NohEleLW", &NohEleLW, &b_NohEleLW);
  fChain->SetBranchAddress("ohEleEtLW", ohEleEtLW, &b_ohEleEtLW);
  fChain->SetBranchAddress("ohEleEtaLW", ohEleEtaLW, &b_ohEleEtaLW);
  fChain->SetBranchAddress("ohElePhiLW", ohElePhiLW, &b_ohElePhiLW);
  fChain->SetBranchAddress("ohEleELW", ohEleELW, &b_ohEleELW);
  fChain->SetBranchAddress("ohElePLW", ohElePLW, &b_ohElePLW);
  fChain->SetBranchAddress("ohEleHisoLW", ohEleHisoLW, &b_ohEleHisoLW);
  fChain->SetBranchAddress("ohEleTisoLW", ohEleTisoLW, &b_ohEleTisoLW);
  fChain->SetBranchAddress("ohEleL1isoLW", ohEleL1isoLW, &b_ohEleL1isoLW);
  fChain->SetBranchAddress("ohElePixelSeedsLW", ohElePixelSeedsLW, &b_ohElePixelSeedsLW);
  fChain->SetBranchAddress("ohEleNewSCLW", ohEleNewSCLW, &b_ohEleNewSCLW);
  fChain->SetBranchAddress("NrecoMuon", &NrecoMuon, &b_NrecoMuon);
  fChain->SetBranchAddress("recoMuonPt", &recoMuonPt, &b_recoMuonPt);
  fChain->SetBranchAddress("recoMuonPhi", &recoMuonPhi, &b_recoMuonPhi);
  fChain->SetBranchAddress("recoMuonEta", &recoMuonEta, &b_recoMuonEta);
  fChain->SetBranchAddress("recoMuonEt", &recoMuonEt, &b_recoMuonEt);
  fChain->SetBranchAddress("recoMuonE", &recoMuonE, &b_recoMuonE);
  fChain->SetBranchAddress("NohMuL2", &NohMuL2, &b_NohMuL2);
  fChain->SetBranchAddress("ohMuL2Pt", ohMuL2Pt, &b_ohMuL2Pt);
  fChain->SetBranchAddress("ohMuL2Phi", ohMuL2Phi, &b_ohMuL2Phi);
  fChain->SetBranchAddress("ohMuL2Eta", ohMuL2Eta, &b_ohMuL2Eta);
  fChain->SetBranchAddress("ohMuL2Chg", ohMuL2Chg, &b_ohMuL2Chg);
  fChain->SetBranchAddress("ohMuL2PtErr", ohMuL2PtErr, &b_ohMuL2PtErr);
  fChain->SetBranchAddress("ohMuL2Iso", ohMuL2Iso, &b_ohMuL2Iso);
  fChain->SetBranchAddress("ohMuL2Dr", ohMuL2Dr, &b_ohMuL2Dr);
  fChain->SetBranchAddress("ohMuL2Dz", ohMuL2Dz, &b_ohMuL2Dz);
  fChain->SetBranchAddress("NohMuL3", &NohMuL3, &b_NohMuL3);
  fChain->SetBranchAddress("ohMuL3Pt", ohMuL3Pt, &b_ohMuL3Pt);
  fChain->SetBranchAddress("ohMuL3Phi", ohMuL3Phi, &b_ohMuL3Phi);
  fChain->SetBranchAddress("ohMuL3Eta", ohMuL3Eta, &b_ohMuL3Eta);
  fChain->SetBranchAddress("ohMuL3Chg", ohMuL3Chg, &b_ohMuL3Chg);
  fChain->SetBranchAddress("ohMuL3PtErr", ohMuL3PtErr, &b_ohMuL3PtErr);
  fChain->SetBranchAddress("ohMuL3Iso", ohMuL3Iso, &b_ohMuL3Iso);
  fChain->SetBranchAddress("ohMuL3Dr", ohMuL3Dr, &b_ohMuL3Dr);
  fChain->SetBranchAddress("ohMuL3Dz", ohMuL3Dz, &b_ohMuL3Dz);
  fChain->SetBranchAddress("ohMuL3L2idx", ohMuL3L2idx, &b_ohMuL3L2idx);
  fChain->SetBranchAddress("NMCpart", &NMCpart, &b_NMCpart);
  fChain->SetBranchAddress("MCpid", MCpid, &b_MCpid);
  fChain->SetBranchAddress("MCstatus", MCstatus, &b_MCstatus);
  fChain->SetBranchAddress("MCvtxX", MCvtxX, &b_MCvtxX);
  fChain->SetBranchAddress("MCvtxY", MCvtxY, &b_MCvtxY);
  fChain->SetBranchAddress("MCvtxZ", MCvtxZ, &b_MCvtxZ);
  fChain->SetBranchAddress("MCpt", MCpt, &b_MCpt);
  fChain->SetBranchAddress("MCeta", MCeta, &b_MCeta);
  fChain->SetBranchAddress("MCphi", MCphi, &b_MCphi);
  fChain->SetBranchAddress("MCPtHat", &MCPtHat, &b_MCPtHat);
  fChain->SetBranchAddress("MCmu3", &MCmu3, &b_MCmu3);
  fChain->SetBranchAddress("MCel3", &MCel3, &b_MCel3);
  fChain->SetBranchAddress("MCbb", &MCbb, &b_MCbb);
  fChain->SetBranchAddress("MCab", &MCab, &b_MCab);
  fChain->SetBranchAddress("MCWenu", &MCWenu, &b_MCWenu);
  fChain->SetBranchAddress("MCWmunu", &MCWmunu, &b_MCmunu);
  fChain->SetBranchAddress("MCZee", &MCZee, &b_MCZee);
  fChain->SetBranchAddress("MCZmumu", &MCZmumu, &b_MCZmumu);
  fChain->SetBranchAddress("MCptEleMax", &MCptEleMax, &b_MCptEleMax);
  fChain->SetBranchAddress("MCptMuMax", &MCptMuMax, &b_MCptMuMax);
  fChain->SetBranchAddress("NL1IsolEm", &NL1IsolEm, &b_NL1IsolEm);
  fChain->SetBranchAddress("L1IsolEmEt", L1IsolEmEt, &b_L1IsolEmEt);
  fChain->SetBranchAddress("L1IsolEmE", L1IsolEmE, &b_L1IsolEmE);
  fChain->SetBranchAddress("L1IsolEmEta", L1IsolEmEta, &b_L1IsolEmEta);
  fChain->SetBranchAddress("L1IsolEmPhi", L1IsolEmPhi, &b_L1IsolEmPhi);
  fChain->SetBranchAddress("NL1NIsolEm", &NL1NIsolEm, &b_NL1NIsolEm);
  fChain->SetBranchAddress("L1NIsolEmEt", L1NIsolEmEt, &b_L1NIsolEmEt);
  fChain->SetBranchAddress("L1NIsolEmE", L1NIsolEmE, &b_L1NIsolEmE);
  fChain->SetBranchAddress("L1NIsolEmEta", L1NIsolEmEta, &b_L1NIsolEmEta);
  fChain->SetBranchAddress("L1NIsolEmPhi", L1NIsolEmPhi, &b_L1NIsolEmPhi);
  fChain->SetBranchAddress("NL1Mu", &NL1Mu, &b_NL1Mu);
  fChain->SetBranchAddress("L1MuPt", L1MuPt, &b_L1MuPt);
  fChain->SetBranchAddress("L1MuE", L1MuE, &b_L1MuE);
  fChain->SetBranchAddress("L1MuEta", L1MuEta, &b_L1MuEta);
  fChain->SetBranchAddress("L1MuPhi", L1MuPhi, &b_L1MuPhi);
  fChain->SetBranchAddress("L1MuIsol", L1MuIsol, &b_L1MuIsol);
  fChain->SetBranchAddress("L1MuMip", L1MuMip, &b_L1MuMip);
  fChain->SetBranchAddress("L1MuFor", L1MuFor, &b_L1MuFor);
  fChain->SetBranchAddress("L1MuRPC", L1MuRPC, &b_L1MuRPC);
  fChain->SetBranchAddress("L1MuQal", L1MuQal, &b_L1MuQal);
  fChain->SetBranchAddress("NL1CenJet", &NL1CenJet, &b_NL1CenJet);
  fChain->SetBranchAddress("L1CenJetEt", L1CenJetEt, &b_L1CenJetEt);
  fChain->SetBranchAddress("L1CenJetE", L1CenJetE, &b_L1CenJetE);
  fChain->SetBranchAddress("L1CenJetEta", L1CenJetEta, &b_L1CenJetEta);
  fChain->SetBranchAddress("L1CenJetPhi", L1CenJetPhi, &b_L1CenJetPhi);
  fChain->SetBranchAddress("NL1ForJet", &NL1ForJet, &b_NL1ForJet);
  fChain->SetBranchAddress("L1ForJetEt", L1ForJetEt, &b_L1ForJetEt);
  fChain->SetBranchAddress("L1ForJetE", L1ForJetE, &b_L1ForJetE);
  fChain->SetBranchAddress("L1ForJetEta", L1ForJetEta, &b_L1ForJetEta);
  fChain->SetBranchAddress("L1ForJetPhi", L1ForJetPhi, &b_L1ForJetPhi);
  fChain->SetBranchAddress("NL1Tau", &NL1Tau, &b_NL1Tau);
  fChain->SetBranchAddress("L1TauEt", L1TauEt, &b_L1TauEt);
  fChain->SetBranchAddress("L1TauE", L1TauE, &b_L1TauE);
  fChain->SetBranchAddress("L1TauEta", L1TauEta, &b_L1TauEta);
  fChain->SetBranchAddress("L1TauPhi", L1TauPhi, &b_L1TauPhi);
  fChain->SetBranchAddress("L1Met", &L1Met, &b_L1Met);
  fChain->SetBranchAddress("L1MetPhi", &L1MetPhi, &b_L1MetPhi);
  fChain->SetBranchAddress("L1MetTot", &L1MetTot, &b_L1MetTot);
  fChain->SetBranchAddress("L1MetHad", &L1MetHad, &b_L1MetHad);
  fChain->SetBranchAddress("L1HfRing0EtSumPositiveEta", &L1HfRing0EtSumPositiveEta, &b_L1HfRing0EtSumPositiveEta);
  fChain->SetBranchAddress("L1HfRing1EtSumPositiveEta", &L1HfRing1EtSumPositiveEta, &b_L1HfRing1EtSumPositiveEta);
  fChain->SetBranchAddress("L1HfRing0EtSumNegativeEta", &L1HfRing0EtSumNegativeEta, &b_L1HfRing0EtSumNegativeEta);
  fChain->SetBranchAddress("L1HfRing1EtSumNegativeEta", &L1HfRing1EtSumNegativeEta, &b_L1HfRing1EtSumNegativeEta);
  fChain->SetBranchAddress("L1HfTowerCountPositiveEta", &L1HfTowerCountPositiveEta, &b_L1HfTowerCountPositiveEta);
  fChain->SetBranchAddress("L1HfTowerCountNegativeEta", &L1HfTowerCountNegativeEta, &b_L1HfTowerCountNegativeEta);
  fChain->SetBranchAddress("Run", &Run, &b_Run);
  fChain->SetBranchAddress("Event", &Event, &b_Event);
  //20X

  //L1's
  fChain->SetBranchAddress("L1_DoubleEG10_00001", &L1_DoubleEG10_00001, &b_L1_DoubleEG10_00001); 
  fChain->SetBranchAddress("L1_DoubleEG1_00001", &L1_DoubleEG1_00001, &b_L1_DoubleEG1_00001); 
  fChain->SetBranchAddress("L1_DoubleEG5_00001", &L1_DoubleEG5_00001, &b_L1_DoubleEG5_00001); 
  fChain->SetBranchAddress("L1_DoubleForJet20", &L1_DoubleForJet20, &b_L1_DoubleForJet20); 
  fChain->SetBranchAddress("L1_DoubleHfBitCountsRing1_P1N1", &L1_DoubleHfBitCountsRing1_P1N1, &b_L1_DoubleHfBitCountsRing1_P1N1); 
  fChain->SetBranchAddress("L1_DoubleHfBitCountsRing2_P1N1", &L1_DoubleHfBitCountsRing2_P1N1, &b_L1_DoubleHfBitCountsRing2_P1N1); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing1_P200N200", &L1_DoubleHfRingEtSumsRing1_P200N200, &b_L1_DoubleHfRingEtSumsRing1_P200N200); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing1_P4N4", &L1_DoubleHfRingEtSumsRing1_P4N4, &b_L1_DoubleHfRingEtSumsRing1_P4N4); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing2_P200N200", &L1_DoubleHfRingEtSumsRing2_P200N200, &b_L1_DoubleHfRingEtSumsRing2_P200N200); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing2_P4N4", &L1_DoubleHfRingEtSumsRing2_P4N4, &b_L1_DoubleHfRingEtSumsRing2_P4N4); 
  fChain->SetBranchAddress("L1_DoubleIsoEG05_TopBottom", &L1_DoubleIsoEG05_TopBottom, &b_L1_DoubleIsoEG05_TopBottom); 
  fChain->SetBranchAddress("L1_DoubleIsoEG05_TopBottomCen", &L1_DoubleIsoEG05_TopBottomCen, &b_L1_DoubleIsoEG05_TopBottomCen); 
  fChain->SetBranchAddress("L1_DoubleIsoEG10_00001", &L1_DoubleIsoEG10_00001, &b_L1_DoubleIsoEG10_00001); 
  fChain->SetBranchAddress("L1_DoubleIsoEG8_00001", &L1_DoubleIsoEG8_00001, &b_L1_DoubleIsoEG8_00001); 
  fChain->SetBranchAddress("L1_DoubleJet40_00001", &L1_DoubleJet40_00001, &b_L1_DoubleJet40_00001); 
  fChain->SetBranchAddress("L1_DoubleJet60_00001", &L1_DoubleJet60_00001, &b_L1_DoubleJet60_00001); 
  fChain->SetBranchAddress("L1_DoubleMu3", &L1_DoubleMu3, &b_L1_DoubleMu3); 
  fChain->SetBranchAddress("L1_DoubleMuOpen", &L1_DoubleMuOpen, &b_L1_DoubleMuOpen); 
  fChain->SetBranchAddress("L1_DoubleMuTopBottom", &L1_DoubleMuTopBottom, &b_L1_DoubleMuTopBottom); 
  fChain->SetBranchAddress("L1_DoubleNoIsoEG05_TopBottom", &L1_DoubleNoIsoEG05_TopBottom, &b_L1_DoubleNoIsoEG05_TopBottom); 
  fChain->SetBranchAddress("L1_DoubleNoIsoEG05_TopBottomCen", &L1_DoubleNoIsoEG05_TopBottomCen, &b_L1_DoubleNoIsoEG05_TopBottomCen); 
  fChain->SetBranchAddress("L1_DoubleTauJet20_00001", &L1_DoubleTauJet20_00001, &b_L1_DoubleTauJet20_00001); 
  fChain->SetBranchAddress("L1_DoubleTauJet8_00001", &L1_DoubleTauJet8_00001, &b_L1_DoubleTauJet8_00001); 
  fChain->SetBranchAddress("L1_EG12_Jet40_00001", &L1_EG12_Jet40_00001, &b_L1_EG12_Jet40_00001); 
  fChain->SetBranchAddress("L1_EG5_TripleJet6_00001", &L1_EG5_TripleJet6_00001, &b_L1_EG5_TripleJet6_00001); 
  fChain->SetBranchAddress("L1_ETM20_00001", &L1_ETM20_00001, &b_L1_ETM20_00001); 
  fChain->SetBranchAddress("L1_ETM30_00001", &L1_ETM30_00001, &b_L1_ETM30_00001); 
  fChain->SetBranchAddress("L1_ETM40_00001", &L1_ETM40_00001, &b_L1_ETM40_00001); 
  fChain->SetBranchAddress("L1_ETM50_00001", &L1_ETM50_00001, &b_L1_ETM50_00001); 
  fChain->SetBranchAddress("L1_ETT60_00001", &L1_ETT60_00001, &b_L1_ETT60_00001); 
  fChain->SetBranchAddress("L1_HTT100_00001", &L1_HTT100_00001, &b_L1_HTT100_00001); 
  fChain->SetBranchAddress("L1_HTT200_00001", &L1_HTT200_00001, &b_L1_HTT200_00001); 
  fChain->SetBranchAddress("L1_HTT300_00001", &L1_HTT300_00001, &b_L1_HTT300_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet12_00001", &L1_IsoEG10_Jet12_00001, &b_L1_IsoEG10_Jet12_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet6_00001", &L1_IsoEG10_Jet6_00001, &b_L1_IsoEG10_Jet6_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet6_ForJet6_00001", &L1_IsoEG10_Jet6_ForJet6_00001, &b_L1_IsoEG10_Jet6_ForJet6_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet8_00001", &L1_IsoEG10_Jet8_00001, &b_L1_IsoEG10_Jet8_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_TauJet8_00001", &L1_IsoEG10_TauJet8_00001, &b_L1_IsoEG10_TauJet8_00001); 
  fChain->SetBranchAddress("L1_MinBias_ETT10_00001", &L1_MinBias_ETT10_00001, &b_L1_MinBias_ETT10_00001); 
  fChain->SetBranchAddress("L1_MinBias_HTT10_00001", &L1_MinBias_HTT10_00001, &b_L1_MinBias_HTT10_00001); 
  fChain->SetBranchAddress("L1_Mu3_EG12_00001", &L1_Mu3_EG12_00001, &b_L1_Mu3_EG12_00001); 
  fChain->SetBranchAddress("L1_Mu3_IsoEG5_00001", &L1_Mu3_IsoEG5_00001, &b_L1_Mu3_IsoEG5_00001); 
  fChain->SetBranchAddress("L1_Mu3_Jet6_00001", &L1_Mu3_Jet6_00001, &b_L1_Mu3_Jet6_00001); 
  fChain->SetBranchAddress("L1_Mu3_TripleJet6_00001", &L1_Mu3_TripleJet6_00001, &b_L1_Mu3_TripleJet6_00001); 
  fChain->SetBranchAddress("L1_Mu5_IsoEG10_00001", &L1_Mu5_IsoEG10_00001, &b_L1_Mu5_IsoEG10_00001); 
  fChain->SetBranchAddress("L1_Mu5_Jet6_00001", &L1_Mu5_Jet6_00001, &b_L1_Mu5_Jet6_00001); 
  fChain->SetBranchAddress("L1_Mu5_TauJet8_00001", &L1_Mu5_TauJet8_00001, &b_L1_Mu5_TauJet8_00001); 
  fChain->SetBranchAddress("L1_QuadJet20_00001", &L1_QuadJet20_00001, &b_L1_QuadJet20_00001); 
  fChain->SetBranchAddress("L1_QuadJet6_00001", &L1_QuadJet6_00001, &b_L1_QuadJet6_00001); 
  fChain->SetBranchAddress("L1_SingleEG1", &L1_SingleEG1, &b_L1_SingleEG1); 
  fChain->SetBranchAddress("L1_SingleEG10_00001", &L1_SingleEG10_00001, &b_L1_SingleEG10_00001); 
  fChain->SetBranchAddress("L1_SingleEG12_00001", &L1_SingleEG12_00001, &b_L1_SingleEG12_00001); 
  fChain->SetBranchAddress("L1_SingleEG15_00001", &L1_SingleEG15_00001, &b_L1_SingleEG15_00001); 
  fChain->SetBranchAddress("L1_SingleEG20_00001", &L1_SingleEG20_00001, &b_L1_SingleEG20_00001); 
  fChain->SetBranchAddress("L1_SingleEG5_00001", &L1_SingleEG5_00001, &b_L1_SingleEG5_00001); 
  fChain->SetBranchAddress("L1_SingleEG5_Endcap_00001", &L1_SingleEG5_Endcap_00001, &b_L1_SingleEG5_Endcap_00001); 
  fChain->SetBranchAddress("L1_SingleEG8_00001", &L1_SingleEG8_00001, &b_L1_SingleEG8_00001); 
  fChain->SetBranchAddress("L1_SingleForJet10", &L1_SingleForJet10, &b_L1_SingleForJet10); 
  fChain->SetBranchAddress("L1_SingleForJet6", &L1_SingleForJet6, &b_L1_SingleForJet6); 
  fChain->SetBranchAddress("L1_SingleHfBitCountsRing1_1", &L1_SingleHfBitCountsRing1_1, &b_L1_SingleHfBitCountsRing1_1); 
  fChain->SetBranchAddress("L1_SingleHfBitCountsRing2_1", &L1_SingleHfBitCountsRing2_1, &b_L1_SingleHfBitCountsRing2_1); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing1_200", &L1_SingleHfRingEtSumsRing1_200, &b_L1_SingleHfRingEtSumsRing1_200); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing1_4", &L1_SingleHfRingEtSumsRing1_4, &b_L1_SingleHfRingEtSumsRing1_4); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing2_200", &L1_SingleHfRingEtSumsRing2_200, &b_L1_SingleHfRingEtSumsRing2_200); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing2_4", &L1_SingleHfRingEtSumsRing2_4, &b_L1_SingleHfRingEtSumsRing2_4); 
  fChain->SetBranchAddress("L1_SingleIsoEG10_00001", &L1_SingleIsoEG10_00001, &b_L1_SingleIsoEG10_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG12_00001", &L1_SingleIsoEG12_00001, &b_L1_SingleIsoEG12_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG15_00001", &L1_SingleIsoEG15_00001, &b_L1_SingleIsoEG15_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG5_00001", &L1_SingleIsoEG5_00001, &b_L1_SingleIsoEG5_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG5_Endcap_00001", &L1_SingleIsoEG5_Endcap_00001, &b_L1_SingleIsoEG5_Endcap_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG8_00001", &L1_SingleIsoEG8_00001, &b_L1_SingleIsoEG8_00001); 
  fChain->SetBranchAddress("L1_SingleJet10_00001", &L1_SingleJet10_00001, &b_L1_SingleJet10_00001); 
  fChain->SetBranchAddress("L1_SingleJet10_Barrel_00001", &L1_SingleJet10_Barrel_00001, &b_L1_SingleJet10_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet10_Central", &L1_SingleJet10_Central, &b_L1_SingleJet10_Central); 
  fChain->SetBranchAddress("L1_SingleJet10_Endcap", &L1_SingleJet10_Endcap, &b_L1_SingleJet10_Endcap); 
  fChain->SetBranchAddress("L1_SingleJet20_00001", &L1_SingleJet20_00001, &b_L1_SingleJet20_00001); 
  fChain->SetBranchAddress("L1_SingleJet20_Barrel_00001", &L1_SingleJet20_Barrel_00001, &b_L1_SingleJet20_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet30_00001", &L1_SingleJet30_00001, &b_L1_SingleJet30_00001); 
  fChain->SetBranchAddress("L1_SingleJet30_Barrel_00001", &L1_SingleJet30_Barrel_00001, &b_L1_SingleJet30_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet40_00001", &L1_SingleJet40_00001, &b_L1_SingleJet40_00001); 
  fChain->SetBranchAddress("L1_SingleJet40_Barrel_00001", &L1_SingleJet40_Barrel_00001, &b_L1_SingleJet40_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet50_00001", &L1_SingleJet50_00001, &b_L1_SingleJet50_00001); 
  fChain->SetBranchAddress("L1_SingleJet60_00001", &L1_SingleJet60_00001, &b_L1_SingleJet60_00001); 
  fChain->SetBranchAddress("L1_SingleJet6_00001", &L1_SingleJet6_00001, &b_L1_SingleJet6_00001); 
  fChain->SetBranchAddress("L1_SingleJet6_Barrel_00001", &L1_SingleJet6_Barrel_00001, &b_L1_SingleJet6_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet6_Central", &L1_SingleJet6_Central, &b_L1_SingleJet6_Central); 
  fChain->SetBranchAddress("L1_SingleJet6_Endcap", &L1_SingleJet6_Endcap, &b_L1_SingleJet6_Endcap); 
  fChain->SetBranchAddress("L1_SingleMu0", &L1_SingleMu0, &b_L1_SingleMu0); 
  fChain->SetBranchAddress("L1_SingleMu10", &L1_SingleMu10, &b_L1_SingleMu10); 
  fChain->SetBranchAddress("L1_SingleMu14", &L1_SingleMu14, &b_L1_SingleMu14); 
  fChain->SetBranchAddress("L1_SingleMu3", &L1_SingleMu3, &b_L1_SingleMu3); 
  fChain->SetBranchAddress("L1_SingleMu5", &L1_SingleMu5, &b_L1_SingleMu5); 
  fChain->SetBranchAddress("L1_SingleMu7", &L1_SingleMu7, &b_L1_SingleMu7); 
  fChain->SetBranchAddress("L1_SingleMuBeamHalo", &L1_SingleMuBeamHalo, &b_L1_SingleMuBeamHalo); 
  fChain->SetBranchAddress("L1_SingleMuOpen", &L1_SingleMuOpen, &b_L1_SingleMuOpen); 
  fChain->SetBranchAddress("L1_SingleTauJet10_00001", &L1_SingleTauJet10_00001, &b_L1_SingleTauJet10_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet10_Barrel_00001", &L1_SingleTauJet10_Barrel_00001, &b_L1_SingleTauJet10_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet20_00001", &L1_SingleTauJet20_00001, &b_L1_SingleTauJet20_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet20_Barrel_00001", &L1_SingleTauJet20_Barrel_00001, &b_L1_SingleTauJet20_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet30_00001", &L1_SingleTauJet30_00001, &b_L1_SingleTauJet30_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet30_Barrel_00001", &L1_SingleTauJet30_Barrel_00001, &b_L1_SingleTauJet30_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet50_00001", &L1_SingleTauJet50_00001, &b_L1_SingleTauJet50_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet8_00001", &L1_SingleTauJet8_00001, &b_L1_SingleTauJet8_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet8_Barrel_00001", &L1_SingleTauJet8_Barrel_00001, &b_L1_SingleTauJet8_Barrel_00001); 
  fChain->SetBranchAddress("L1_TauJet10_ETM30_00001", &L1_TauJet10_ETM30_00001, &b_L1_TauJet10_ETM30_00001); 
  fChain->SetBranchAddress("L1_TauJet10_ETM40_00001", &L1_TauJet10_ETM40_00001, &b_L1_TauJet10_ETM40_00001); 
  fChain->SetBranchAddress("L1_TripleJet30_00001", &L1_TripleJet30_00001, &b_L1_TripleJet30_00001); 
  fChain->SetBranchAddress("L1_TripleMu3", &L1_TripleMu3, &b_L1_TripleMu3); 

  // 21X HLT names
  fChain->SetBranchAddress("HLT_L1Jet15", &HLT_L1Jet15, &b_HLT_L1Jet15);
  fChain->SetBranchAddress("HLT_Jet30", &HLT_Jet30, &b_HLT_Jet30);
  fChain->SetBranchAddress("HLT_Jet50", &HLT_Jet50, &b_HLT_Jet50);
  fChain->SetBranchAddress("HLT_Jet80", &HLT_Jet80, &b_HLT_Jet80);
  fChain->SetBranchAddress("HLT_Jet110", &HLT_Jet110, &b_HLT_Jet110);
  fChain->SetBranchAddress("HLT_Jet180", &HLT_Jet180, &b_HLT_Jet180);
  fChain->SetBranchAddress("HLT_Jet250", &HLT_Jet250, &b_HLT_Jet250);
  fChain->SetBranchAddress("HLT_FwdJet20", &HLT_FwdJet20, &b_HLT_FwdJet20);
  fChain->SetBranchAddress("HLT_DoubleJet150", &HLT_DoubleJet150, &b_HLT_DoubleJet150);
  fChain->SetBranchAddress("HLT_DoubleJet125_Aco", &HLT_DoubleJet125_Aco, &b_HLT_DoubleJet125_Aco);
  fChain->SetBranchAddress("HLT_DoubleFwdJet50", &HLT_DoubleFwdJet50, &b_HLT_DoubleFwdJet50);
  fChain->SetBranchAddress("HLT_DiJetAve15", &HLT_DiJetAve15, &b_HLT_DiJetAve15);
  fChain->SetBranchAddress("HLT_DiJetAve30", &HLT_DiJetAve30, &b_HLT_DiJetAve30);
  fChain->SetBranchAddress("HLT_DiJetAve50", &HLT_DiJetAve50, &b_HLT_DiJetAve50);
  fChain->SetBranchAddress("HLT_DiJetAve70", &HLT_DiJetAve70, &b_HLT_DiJetAve70);
  fChain->SetBranchAddress("HLT_DiJetAve130", &HLT_DiJetAve130, &b_HLT_DiJetAve130);
  fChain->SetBranchAddress("HLT_DiJetAve220", &HLT_DiJetAve220, &b_HLT_DiJetAve220);
  fChain->SetBranchAddress("HLT_TripleJet85", &HLT_TripleJet85, &b_HLT_TripleJet85);
  fChain->SetBranchAddress("HLT_QuadJet30", &HLT_QuadJet30, &b_HLT_QuadJet30);
  fChain->SetBranchAddress("HLT_QuadJet60", &HLT_QuadJet60, &b_HLT_QuadJet60);
  fChain->SetBranchAddress("HLT_SumET120", &HLT_SumET120, &b_HLT_SumET120);
  fChain->SetBranchAddress("HLT_L1MET20", &HLT_L1MET20, &b_HLT_L1MET20);
  fChain->SetBranchAddress("HLT_MET25", &HLT_MET25, &b_HLT_MET25);
  fChain->SetBranchAddress("HLT_MET35", &HLT_MET35, &b_HLT_MET35);
  fChain->SetBranchAddress("HLT_MET50", &HLT_MET50, &b_HLT_MET50);
  fChain->SetBranchAddress("HLT_MET65", &HLT_MET65, &b_HLT_MET65);
  fChain->SetBranchAddress("HLT_MET75", &HLT_MET75, &b_HLT_MET75);
  fChain->SetBranchAddress("HLT_MET35_HT350", &HLT_MET35_HT350, &b_HLT_MET35_HT350);
  fChain->SetBranchAddress("HLT_Jet180_MET60", &HLT_Jet180_MET60, &b_HLT_Jet180_MET60);
  fChain->SetBranchAddress("HLT_Jet60_MET70_Aco", &HLT_Jet60_MET70_Aco, &b_HLT_Jet60_MET70_Aco);
  fChain->SetBranchAddress("HLT_Jet100_MET60_Aco", &HLT_Jet100_MET60_Aco, &b_HLT_Jet100_MET60_Aco);
  fChain->SetBranchAddress("HLT_DoubleJet125_MET60", &HLT_DoubleJet125_MET60, &b_HLT_DoubleJet125_MET60);
  fChain->SetBranchAddress("HLT_DoubleFwdJet40_MET60", &HLT_DoubleFwdJet40_MET60, &b_HLT_DoubleFwdJet40_MET60);
  fChain->SetBranchAddress("HLT_DoubleJet60_MET60_Aco", &HLT_DoubleJet60_MET60_Aco, &b_HLT_DoubleJet60_MET60_Aco);
  fChain->SetBranchAddress("HLT_DoubleJet50_MET70_Aco", &HLT_DoubleJet50_MET70_Aco, &b_HLT_DoubleJet50_MET70_Aco);
  fChain->SetBranchAddress("HLT_DoubleJet40_MET70_Aco", &HLT_DoubleJet40_MET70_Aco, &b_HLT_DoubleJet40_MET70_Aco);
  fChain->SetBranchAddress("HLT_TripleJet60_MET60", &HLT_TripleJet60_MET60, &b_HLT_TripleJet60_MET60);
  fChain->SetBranchAddress("HLT_QuadJet35_MET60", &HLT_QuadJet35_MET60, &b_HLT_QuadJet35_MET60);
  fChain->SetBranchAddress("HLT_IsoEle15_L1I", &HLT_IsoEle15_L1I, &b_HLT_IsoEle15_L1I);
  fChain->SetBranchAddress("HLT_IsoEle18_L1R", &HLT_IsoEle18_L1R, &b_HLT_IsoEle18_L1R);
  fChain->SetBranchAddress("HLT_IsoEle15_LW_L1I", &HLT_IsoEle15_LW_L1I, &b_HLT_IsoEle15_LW_L1I);
  fChain->SetBranchAddress("HLT_LooseIsoEle15_LW_L1R", &HLT_LooseIsoEle15_LW_L1R, &b_HLT_LooseIsoEle15_LW_L1R);
  fChain->SetBranchAddress("HLT_Ele10_SW_L1R", &HLT_Ele10_SW_L1R, &b_HLT_Ele10_SW_L1R);
  fChain->SetBranchAddress("HLT_Ele15_SW_L1R", &HLT_Ele15_SW_L1R, &b_HLT_Ele15_SW_L1R);
  fChain->SetBranchAddress("HLT_Ele15_LW_L1R", &HLT_Ele15_LW_L1R, &b_HLT_Ele15_LW_L1R);
  fChain->SetBranchAddress("HLT_EM80", &HLT_EM80, &b_HLT_EM80);
  fChain->SetBranchAddress("HLT_EM200", &HLT_EM200, &b_HLT_EM200);
  fChain->SetBranchAddress("HLT_DoubleIsoEle10_L1I", &HLT_DoubleIsoEle10_L1I, &b_HLT_DoubleIsoEle10_L1I);
  fChain->SetBranchAddress("HLT_DoubleIsoEle12_L1R", &HLT_DoubleIsoEle12_L1R, &b_HLT_DoubleIsoEle12_L1R);
  fChain->SetBranchAddress("HLT_DoubleIsoEle10_LW_L1I", &HLT_DoubleIsoEle10_LW_L1I, &b_HLT_DoubleIsoEle10_LW_L1I);
  fChain->SetBranchAddress("HLT_DoubleIsoEle12_LW_L1R", &HLT_DoubleIsoEle12_LW_L1R, &b_HLT_DoubleIsoEle12_LW_L1R);
  fChain->SetBranchAddress("HLT_DoubleEle5_SW_L1R", &HLT_DoubleEle5_SW_L1R, &b_HLT_DoubleEle5_SW_L1R);
  fChain->SetBranchAddress("HLT_DoubleEle10_LW_OnlyPixelM_L1R", &HLT_DoubleEle10_LW_OnlyPixelM_L1R, &b_HLT_DoubleEle10_LW_OnlyPixelM_L1R);
  fChain->SetBranchAddress("HLT_DoubleEle10_Z", &HLT_DoubleEle10_Z, &b_HLT_DoubleEle10_Z);
  fChain->SetBranchAddress("HLT_DoubleEle6_Exclusive", &HLT_DoubleEle6_Exclusive, &b_HLT_DoubleEle6_Exclusive);
  fChain->SetBranchAddress("HLT_IsoPhoton30_L1I", &HLT_IsoPhoton30_L1I, &b_HLT_IsoPhoton30_L1I);
  fChain->SetBranchAddress("HLT_IsoPhoton10_L1R", &HLT_IsoPhoton10_L1R, &b_HLT_IsoPhoton10_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton15_L1R", &HLT_IsoPhoton15_L1R, &b_HLT_IsoPhoton15_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton20_L1R", &HLT_IsoPhoton20_L1R, &b_HLT_IsoPhoton20_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton25_L1R", &HLT_IsoPhoton25_L1R, &b_HLT_IsoPhoton25_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton40_L1R", &HLT_IsoPhoton40_L1R, &b_HLT_IsoPhoton40_L1R);
  fChain->SetBranchAddress("HLT_Photon15_L1R", &HLT_Photon15_L1R, &b_HLT_Photon15_L1R);
  fChain->SetBranchAddress("HLT_Photon25_L1R", &HLT_Photon25_L1R, &b_HLT_Photon25_L1R);
  fChain->SetBranchAddress("HLT_DoubleIsoPhoton20_L1I", &HLT_DoubleIsoPhoton20_L1I, &b_HLT_DoubleIsoPhoton20_L1I);
  fChain->SetBranchAddress("HLT_DoubleIsoPhoton20_L1R", &HLT_DoubleIsoPhoton20_L1R, &b_HLT_DoubleIsoPhoton20_L1R);
  fChain->SetBranchAddress("HLT_DoublePhoton10_Exclusive", &HLT_DoublePhoton10_Exclusive, &b_HLT_DoublePhoton10_Exclusive);
  fChain->SetBranchAddress("HLT_L1Mu", &HLT_L1Mu, &b_HLT_L1Mu);
  fChain->SetBranchAddress("HLT_L1MuOpen", &HLT_L1MuOpen, &b_HLT_L1MuOpen);
  fChain->SetBranchAddress("HLT_L2Mu9", &HLT_L2Mu9, &b_HLT_L2Mu9);
  fChain->SetBranchAddress("HLT_IsoMu9", &HLT_IsoMu9, &b_HLT_IsoMu9);
  fChain->SetBranchAddress("HLT_IsoMu11", &HLT_IsoMu11, &b_HLT_IsoMu11);
  fChain->SetBranchAddress("HLT_IsoMu13", &HLT_IsoMu13, &b_HLT_IsoMu13);
  fChain->SetBranchAddress("HLT_IsoMu15", &HLT_IsoMu15, &b_HLT_IsoMu15);
  fChain->SetBranchAddress("HLT_NoTrackerIsoMu15", &HLT_NoTrackerIsoMu15, &b_HLT_NoTrackerIsoMu15);
  fChain->SetBranchAddress("HLT_Mu3", &HLT_Mu3, &b_HLT_Mu3);
  fChain->SetBranchAddress("HLT_Mu5", &HLT_Mu5, &b_HLT_Mu5);
  fChain->SetBranchAddress("HLT_Mu7", &HLT_Mu7, &b_HLT_Mu7);
  fChain->SetBranchAddress("HLT_Mu9", &HLT_Mu9, &b_HLT_Mu9);
  fChain->SetBranchAddress("HLT_Mu11", &HLT_Mu11, &b_HLT_Mu11);
  fChain->SetBranchAddress("HLT_Mu13", &HLT_Mu13, &b_HLT_Mu13);
  fChain->SetBranchAddress("HLT_Mu15", &HLT_Mu15, &b_HLT_Mu15);
  fChain->SetBranchAddress("HLT_Mu15_L1Mu7", &HLT_Mu15_L1Mu7, &b_HLT_Mu15_L1Mu7);
  fChain->SetBranchAddress("HLT_Mu15_Vtx2cm", &HLT_Mu15_Vtx2cm, &b_HLT_Mu15_Vtx2cm);
  fChain->SetBranchAddress("HLT_Mu15_Vtx2mm", &HLT_Mu15_Vtx2mm, &b_HLT_Mu15_Vtx2mm);
  fChain->SetBranchAddress("HLT_DoubleIsoMu3", &HLT_DoubleIsoMu3, &b_HLT_DoubleIsoMu3);
  fChain->SetBranchAddress("HLT_DoubleMu3", &HLT_DoubleMu3, &b_HLT_DoubleMu3);
  fChain->SetBranchAddress("HLT_DoubleMu3_Vtx2cm", &HLT_DoubleMu3_Vtx2cm, &b_HLT_DoubleMu3_Vtx2cm);
  fChain->SetBranchAddress("HLT_DoubleMu3_Vtx2mm", &HLT_DoubleMu3_Vtx2mm, &b_HLT_DoubleMu3_Vtx2mm);
  fChain->SetBranchAddress("HLT_DoubleMu3_JPsi", &HLT_DoubleMu3_JPsi, &b_HLT_DoubleMu3_JPsi);
  fChain->SetBranchAddress("HLT_DoubleMu3_Upsilon", &HLT_DoubleMu3_Upsilon, &b_HLT_DoubleMu3_Upsilon);
  fChain->SetBranchAddress("HLT_DoubleMu7_Z", &HLT_DoubleMu7_Z, &b_HLT_DoubleMu7_Z);
  fChain->SetBranchAddress("HLT_DoubleMu3_SameSign", &HLT_DoubleMu3_SameSign, &b_HLT_DoubleMu3_SameSign);
  fChain->SetBranchAddress("HLT_DoubleMu3_Psi2S", &HLT_DoubleMu3_Psi2S, &b_HLT_DoubleMu3_Psi2S);
  fChain->SetBranchAddress("HLT_BTagIP_Jet180", &HLT_BTagIP_Jet180, &b_HLT_BTagIP_Jet180);
  fChain->SetBranchAddress("HLT_BTagIP_Jet120_Relaxed", &HLT_BTagIP_Jet120_Relaxed, &b_HLT_BTagIP_Jet120_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_DoubleJet120", &HLT_BTagIP_DoubleJet120, &b_HLT_BTagIP_DoubleJet120);
  fChain->SetBranchAddress("HLT_BTagIP_DoubleJet60_Relaxed", &HLT_BTagIP_DoubleJet60_Relaxed, &b_HLT_BTagIP_DoubleJet60_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_TripleJet70", &HLT_BTagIP_TripleJet70, &b_HLT_BTagIP_TripleJet70);
  fChain->SetBranchAddress("HLT_BTagIP_TripleJet40_Relaxed", &HLT_BTagIP_TripleJet40_Relaxed, &b_HLT_BTagIP_TripleJet40_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_QuadJet40", &HLT_BTagIP_QuadJet40, &b_HLT_BTagIP_QuadJet40);
  fChain->SetBranchAddress("HLT_BTagIP_QuadJet30_Relaxed", &HLT_BTagIP_QuadJet30_Relaxed, &b_HLT_BTagIP_QuadJet30_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_HT470", &HLT_BTagIP_HT470, &b_HLT_BTagIP_HT470);
  fChain->SetBranchAddress("HLT_BTagIP_HT320_Relaxed", &HLT_BTagIP_HT320_Relaxed, &b_HLT_BTagIP_HT320_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_DoubleJet120", &HLT_BTagMu_DoubleJet120, &b_HLT_BTagMu_DoubleJet120);
  fChain->SetBranchAddress("HLT_BTagMu_DoubleJet60_Relaxed", &HLT_BTagMu_DoubleJet60_Relaxed, &b_HLT_BTagMu_DoubleJet60_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_TripleJet70", &HLT_BTagMu_TripleJet70, &b_HLT_BTagMu_TripleJet70);
  fChain->SetBranchAddress("HLT_BTagMu_TripleJet40_Relaxed", &HLT_BTagMu_TripleJet40_Relaxed, &b_HLT_BTagMu_TripleJet40_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_QuadJet40", &HLT_BTagMu_QuadJet40, &b_HLT_BTagMu_QuadJet40);
  fChain->SetBranchAddress("HLT_BTagMu_QuadJet30_Relaxed", &HLT_BTagMu_QuadJet30_Relaxed, &b_HLT_BTagMu_QuadJet30_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_HT370", &HLT_BTagMu_HT370, &b_HLT_BTagMu_HT370);
  fChain->SetBranchAddress("HLT_BTagMu_HT250_Relaxed", &HLT_BTagMu_HT250_Relaxed, &b_HLT_BTagMu_HT250_Relaxed);
  fChain->SetBranchAddress("HLT_DoubleMu3_BJPsi", &HLT_DoubleMu3_BJPsi, &b_HLT_DoubleMu3_BJPsi);
  fChain->SetBranchAddress("HLT_DoubleMu4_BJPsi", &HLT_DoubleMu4_BJPsi, &b_HLT_DoubleMu4_BJPsi);
  fChain->SetBranchAddress("HLT_TripleMu3_TauTo3Mu", &HLT_TripleMu3_TauTo3Mu, &b_HLT_TripleMu3_TauTo3Mu);
  fChain->SetBranchAddress("HLT_IsoTau_MET65_Trk20", &HLT_IsoTau_MET65_Trk20, &b_HLT_IsoTau_MET65_Trk20);
  fChain->SetBranchAddress("HLT_IsoTau_MET35_Trk15_L1MET", &HLT_IsoTau_MET35_Trk15_L1MET, &b_HLT_IsoTau_MET35_Trk15_L1MET);
  fChain->SetBranchAddress("HLT_LooseIsoTau_MET30", &HLT_LooseIsoTau_MET30, &b_HLT_LooseIsoTau_MET30);
  fChain->SetBranchAddress("HLT_LooseIsoTau_MET30_L1MET", &HLT_LooseIsoTau_MET30_L1MET, &b_HLT_LooseIsoTau_MET30_L1MET);
  fChain->SetBranchAddress("HLT_DoubleIsoTau_Trk3", &HLT_DoubleIsoTau_Trk3, &b_HLT_DoubleIsoTau_Trk3);
  fChain->SetBranchAddress("HLT_DoubleLooseIsoTau", &HLT_DoubleLooseIsoTau, &b_HLT_DoubleLooseIsoTau);
  fChain->SetBranchAddress("HLT_IsoEle8_IsoMu7", &HLT_IsoEle8_IsoMu7, &b_HLT_IsoEle8_IsoMu7);
  fChain->SetBranchAddress("HLT_IsoEle10_Mu10_L1R", &HLT_IsoEle10_Mu10_L1R, &b_HLT_IsoEle10_Mu10_L1R);
  fChain->SetBranchAddress("HLT_IsoEle12_IsoTau_Trk3", &HLT_IsoEle12_IsoTau_Trk3, &b_HLT_IsoEle12_IsoTau_Trk3);
  fChain->SetBranchAddress("HLT_IsoEle10_BTagIP_Jet35", &HLT_IsoEle10_BTagIP_Jet35, &b_HLT_IsoEle10_BTagIP_Jet35);
  fChain->SetBranchAddress("HLT_IsoEle12_Jet40", &HLT_IsoEle12_Jet40, &b_HLT_IsoEle12_Jet40);
  fChain->SetBranchAddress("HLT_IsoEle12_DoubleJet80", &HLT_IsoEle12_DoubleJet80, &b_HLT_IsoEle12_DoubleJet80);
  fChain->SetBranchAddress("HLT_IsoElec5_TripleJet30", &HLT_IsoElec5_TripleJet30, &b_HLT_IsoElec5_TripleJet30);
  fChain->SetBranchAddress("HLT_IsoEle12_TripleJet60", &HLT_IsoEle12_TripleJet60, &b_HLT_IsoEle12_TripleJet60);
  fChain->SetBranchAddress("HLT_IsoEle12_QuadJet35", &HLT_IsoEle12_QuadJet35, &b_HLT_IsoEle12_QuadJet35);
  fChain->SetBranchAddress("HLT_IsoMu14_IsoTau_Trk3", &HLT_IsoMu14_IsoTau_Trk3, &b_HLT_IsoMu14_IsoTau_Trk3);
  fChain->SetBranchAddress("HLT_IsoMu7_BTagIP_Jet35", &HLT_IsoMu7_BTagIP_Jet35, &b_HLT_IsoMu7_BTagIP_Jet35);
  fChain->SetBranchAddress("HLT_IsoMu7_BTagMu_Jet20", &HLT_IsoMu7_BTagMu_Jet20, &b_HLT_IsoMu7_BTagMu_Jet20);
  fChain->SetBranchAddress("HLT_IsoMu7_Jet40", &HLT_IsoMu7_Jet40, &b_HLT_IsoMu7_Jet40);
  fChain->SetBranchAddress("HLT_NoL2IsoMu8_Jet40", &HLT_NoL2IsoMu8_Jet40, &b_HLT_NoL2IsoMu8_Jet40);
  fChain->SetBranchAddress("HLT_Mu14_Jet50", &HLT_Mu14_Jet50, &b_HLT_Mu14_Jet50);
  fChain->SetBranchAddress("HLT_Mu5_TripleJet30", &HLT_Mu5_TripleJet30, &b_HLT_Mu5_TripleJet30);
  fChain->SetBranchAddress("HLT_BTagMu_Jet20_Calib", &HLT_BTagMu_Jet20_Calib, &b_HLT_BTagMu_Jet20_Calib);
  fChain->SetBranchAddress("HLT_ZeroBias", &HLT_ZeroBias, &b_HLT_ZeroBias);
  fChain->SetBranchAddress("HLT_MinBias", &HLT_MinBias, &b_HLT_MinBias);
  fChain->SetBranchAddress("HLT_MinBiasHcal", &HLT_MinBiasHcal, &b_HLT_MinBiasHcal);
  fChain->SetBranchAddress("HLT_MinBiasEcal", &HLT_MinBiasEcal, &b_HLT_MinBiasEcal);
  fChain->SetBranchAddress("HLT_MinBiasPixel", &HLT_MinBiasPixel, &b_HLT_MinBiasPixel);
  fChain->SetBranchAddress("HLT_MinBiasPixel_Trk5", &HLT_MinBiasPixel_Trk5, &b_HLT_MinBiasPixel_Trk5);
  fChain->SetBranchAddress("HLT_BackwardBSC", &HLT_BackwardBSC, &b_HLT_BackwardBSC);
  fChain->SetBranchAddress("HLT_ForwardBSC", &HLT_ForwardBSC, &b_HLT_ForwardBSC);
  fChain->SetBranchAddress("HLT_CSCBeamHalo", &HLT_CSCBeamHalo, &b_HLT_CSCBeamHalo);
  fChain->SetBranchAddress("HLT_CSCBeamHaloOverlapRing1", &HLT_CSCBeamHaloOverlapRing1, &b_HLT_CSCBeamHaloOverlapRing1);
  fChain->SetBranchAddress("HLT_CSCBeamHaloOverlapRing2", &HLT_CSCBeamHaloOverlapRing2, &b_HLT_CSCBeamHaloOverlapRing2);
  fChain->SetBranchAddress("HLT_CSCBeamHaloRing2or3", &HLT_CSCBeamHaloRing2or3, &b_HLT_CSCBeamHaloRing2or3);
  fChain->SetBranchAddress("HLT_TrackerCosmics", &HLT_TrackerCosmics, &b_HLT_TrackerCosmics);
  fChain->SetBranchAddress("HLT_TriggerType", &HLT_TriggerType, &b_HLT_TriggerType);
  fChain->SetBranchAddress("AlCa_IsoTrack", &AlCa_IsoTrack, &b_AlCa_IsoTrack);
  fChain->SetBranchAddress("AlCa_EcalPhiSym", &AlCa_EcalPhiSym, &b_AlCa_EcalPhiSym);
  fChain->SetBranchAddress("AlCa_EcalPi0", &AlCa_EcalPi0, &b_AlCa_EcalPi0);

  //
  /* Also associate with the maps to speed up code! */
  fChain->SetBranchAddress("L1_DoubleEG10_00001", &map_BitOfStandardHLTPath["L1_DoubleEG10_00001"], &b_L1_DoubleEG10_00001); 
  fChain->SetBranchAddress("L1_DoubleEG1_00001", &map_BitOfStandardHLTPath["L1_DoubleEG1_00001"], &b_L1_DoubleEG1_00001); 
  fChain->SetBranchAddress("L1_DoubleEG5_00001", &map_BitOfStandardHLTPath["L1_DoubleEG5_00001"], &b_L1_DoubleEG5_00001); 
  fChain->SetBranchAddress("L1_DoubleForJet20", &map_BitOfStandardHLTPath["L1_DoubleForJet20"], &b_L1_DoubleForJet20); 
  fChain->SetBranchAddress("L1_DoubleHfBitCountsRing1_P1N1", &map_BitOfStandardHLTPath["L1_DoubleHfBitCountsRing1_P1N1"], &b_L1_DoubleHfBitCountsRing1_P1N1); 
  fChain->SetBranchAddress("L1_DoubleHfBitCountsRing2_P1N1", &map_BitOfStandardHLTPath["L1_DoubleHfBitCountsRing2_P1N1"], &b_L1_DoubleHfBitCountsRing2_P1N1); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing1_P200N200", &map_BitOfStandardHLTPath["L1_DoubleHfRingEtSumsRing1_P200N200"], &b_L1_DoubleHfRingEtSumsRing1_P200N200); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing1_P4N4", &map_BitOfStandardHLTPath["L1_DoubleHfRingEtSumsRing1_P4N4"], &b_L1_DoubleHfRingEtSumsRing1_P4N4); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing2_P200N200", &map_BitOfStandardHLTPath["L1_DoubleHfRingEtSumsRing2_P200N200"], &b_L1_DoubleHfRingEtSumsRing2_P200N200); 
  fChain->SetBranchAddress("L1_DoubleHfRingEtSumsRing2_P4N4", &map_BitOfStandardHLTPath["L1_DoubleHfRingEtSumsRing2_P4N4"], &b_L1_DoubleHfRingEtSumsRing2_P4N4); 
  fChain->SetBranchAddress("L1_DoubleIsoEG05_TopBottom", &map_BitOfStandardHLTPath["L1_DoubleIsoEG05_TopBottom"], &b_L1_DoubleIsoEG05_TopBottom); 
  fChain->SetBranchAddress("L1_DoubleIsoEG05_TopBottomCen", &map_BitOfStandardHLTPath["L1_DoubleIsoEG05_TopBottomCen"], &b_L1_DoubleIsoEG05_TopBottomCen); 
  fChain->SetBranchAddress("L1_DoubleIsoEG10_00001", &map_BitOfStandardHLTPath["L1_DoubleIsoEG10_00001"], &b_L1_DoubleIsoEG10_00001); 
  fChain->SetBranchAddress("L1_DoubleIsoEG8_00001", &map_BitOfStandardHLTPath["L1_DoubleIsoEG8_00001"], &b_L1_DoubleIsoEG8_00001); 
  fChain->SetBranchAddress("L1_DoubleJet40_00001", &map_BitOfStandardHLTPath["L1_DoubleJet40_00001"], &b_L1_DoubleJet40_00001); 
  fChain->SetBranchAddress("L1_DoubleJet60_00001", &map_BitOfStandardHLTPath["L1_DoubleJet60_00001"], &b_L1_DoubleJet60_00001); 
  fChain->SetBranchAddress("L1_DoubleMu3", &map_BitOfStandardHLTPath["L1_DoubleMu3"], &b_L1_DoubleMu3); 
  fChain->SetBranchAddress("L1_DoubleMuOpen", &map_BitOfStandardHLTPath["L1_DoubleMuOpen"], &b_L1_DoubleMuOpen); 
  fChain->SetBranchAddress("L1_DoubleMuTopBottom", &map_BitOfStandardHLTPath["L1_DoubleMuTopBottom"], &b_L1_DoubleMuTopBottom); 
  fChain->SetBranchAddress("L1_DoubleNoIsoEG05_TopBottom", &map_BitOfStandardHLTPath["L1_DoubleNoIsoEG05_TopBottom"], &b_L1_DoubleNoIsoEG05_TopBottom); 
  fChain->SetBranchAddress("L1_DoubleNoIsoEG05_TopBottomCen", &map_BitOfStandardHLTPath["L1_DoubleNoIsoEG05_TopBottomCen"], &b_L1_DoubleNoIsoEG05_TopBottomCen); 
  fChain->SetBranchAddress("L1_DoubleTauJet20_00001", &map_BitOfStandardHLTPath["L1_DoubleTauJet20_00001"], &b_L1_DoubleTauJet20_00001); 
  fChain->SetBranchAddress("L1_DoubleTauJet8_00001", &map_BitOfStandardHLTPath["L1_DoubleTauJet8_00001"], &b_L1_DoubleTauJet8_00001); 
  fChain->SetBranchAddress("L1_EG12_Jet40_00001", &map_BitOfStandardHLTPath["L1_EG12_Jet40_00001"], &b_L1_EG12_Jet40_00001); 
  fChain->SetBranchAddress("L1_EG5_TripleJet6_00001", &map_BitOfStandardHLTPath["L1_EG5_TripleJet6_00001"], &b_L1_EG5_TripleJet6_00001); 
  fChain->SetBranchAddress("L1_ETM20_00001", &map_BitOfStandardHLTPath["L1_ETM20_00001"], &b_L1_ETM20_00001); 
  fChain->SetBranchAddress("L1_ETM30_00001", &map_BitOfStandardHLTPath["L1_ETM30_00001"], &b_L1_ETM30_00001); 
  fChain->SetBranchAddress("L1_ETM40_00001", &map_BitOfStandardHLTPath["L1_ETM40_00001"], &b_L1_ETM40_00001); 
  fChain->SetBranchAddress("L1_ETM50_00001", &map_BitOfStandardHLTPath["L1_ETM50_00001"], &b_L1_ETM50_00001); 
  fChain->SetBranchAddress("L1_ETT60_00001", &map_BitOfStandardHLTPath["L1_ETT60_00001"], &b_L1_ETT60_00001); 
  fChain->SetBranchAddress("L1_HTT100_00001", &map_BitOfStandardHLTPath["L1_HTT100_00001"], &b_L1_HTT100_00001); 
  fChain->SetBranchAddress("L1_HTT200_00001", &map_BitOfStandardHLTPath["L1_HTT200_00001"], &b_L1_HTT200_00001); 
  fChain->SetBranchAddress("L1_HTT300_00001", &map_BitOfStandardHLTPath["L1_HTT300_00001"], &b_L1_HTT300_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet12_00001", &map_BitOfStandardHLTPath["L1_IsoEG10_Jet12_00001"], &b_L1_IsoEG10_Jet12_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet6_00001", &map_BitOfStandardHLTPath["L1_IsoEG10_Jet6_00001"], &b_L1_IsoEG10_Jet6_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet6_ForJet6_00001", &map_BitOfStandardHLTPath["L1_IsoEG10_Jet6_ForJet6_00001"], &b_L1_IsoEG10_Jet6_ForJet6_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_Jet8_00001", &map_BitOfStandardHLTPath["L1_IsoEG10_Jet8_00001"], &b_L1_IsoEG10_Jet8_00001); 
  fChain->SetBranchAddress("L1_IsoEG10_TauJet8_00001", &map_BitOfStandardHLTPath["L1_IsoEG10_TauJet8_00001"], &b_L1_IsoEG10_TauJet8_00001); 
  fChain->SetBranchAddress("L1_MinBias_ETT10_00001", &map_BitOfStandardHLTPath["L1_MinBias_ETT10_00001"], &b_L1_MinBias_ETT10_00001); 
  fChain->SetBranchAddress("L1_MinBias_HTT10_00001", &map_BitOfStandardHLTPath["L1_MinBias_HTT10_00001"], &b_L1_MinBias_HTT10_00001); 
  fChain->SetBranchAddress("L1_Mu3_EG12_00001", &map_BitOfStandardHLTPath["L1_Mu3_EG12_00001"], &b_L1_Mu3_EG12_00001); 
  fChain->SetBranchAddress("L1_Mu3_IsoEG5_00001", &map_BitOfStandardHLTPath["L1_Mu3_IsoEG5_00001"], &b_L1_Mu3_IsoEG5_00001); 
  fChain->SetBranchAddress("L1_Mu3_Jet6_00001", &map_BitOfStandardHLTPath["L1_Mu3_Jet6_00001"], &b_L1_Mu3_Jet6_00001); 
  fChain->SetBranchAddress("L1_Mu3_TripleJet6_00001", &map_BitOfStandardHLTPath["L1_Mu3_TripleJet6_00001"], &b_L1_Mu3_TripleJet6_00001); 
  fChain->SetBranchAddress("L1_Mu5_IsoEG10_00001", &map_BitOfStandardHLTPath["L1_Mu5_IsoEG10_00001"], &b_L1_Mu5_IsoEG10_00001); 
  fChain->SetBranchAddress("L1_Mu5_Jet6_00001", &map_BitOfStandardHLTPath["L1_Mu5_Jet6_00001"], &b_L1_Mu5_Jet6_00001); 
  fChain->SetBranchAddress("L1_Mu5_TauJet8_00001", &map_BitOfStandardHLTPath["L1_Mu5_TauJet8_00001"], &b_L1_Mu5_TauJet8_00001); 
  fChain->SetBranchAddress("L1_QuadJet20_00001", &map_BitOfStandardHLTPath["L1_QuadJet20_00001"], &b_L1_QuadJet20_00001); 
  fChain->SetBranchAddress("L1_QuadJet6_00001", &map_BitOfStandardHLTPath["L1_QuadJet6_00001"], &b_L1_QuadJet6_00001); 
  fChain->SetBranchAddress("L1_SingleEG1", &map_BitOfStandardHLTPath["L1_SingleEG1"], &b_L1_SingleEG1); 
  fChain->SetBranchAddress("L1_SingleEG10_00001", &map_BitOfStandardHLTPath["L1_SingleEG10_00001"], &b_L1_SingleEG10_00001); 
  fChain->SetBranchAddress("L1_SingleEG12_00001", &map_BitOfStandardHLTPath["L1_SingleEG12_00001"], &b_L1_SingleEG12_00001); 
  fChain->SetBranchAddress("L1_SingleEG15_00001", &map_BitOfStandardHLTPath["L1_SingleEG15_00001"], &b_L1_SingleEG15_00001); 
  fChain->SetBranchAddress("L1_SingleEG20_00001", &map_BitOfStandardHLTPath["L1_SingleEG20_00001"], &b_L1_SingleEG20_00001); 
  fChain->SetBranchAddress("L1_SingleEG5_00001", &map_BitOfStandardHLTPath["L1_SingleEG5_00001"], &b_L1_SingleEG5_00001); 
  fChain->SetBranchAddress("L1_SingleEG5_Endcap_00001", &map_BitOfStandardHLTPath["L1_SingleEG5_Endcap_00001"], &b_L1_SingleEG5_Endcap_00001); 
  fChain->SetBranchAddress("L1_SingleEG8_00001", &map_BitOfStandardHLTPath["L1_SingleEG8_00001"], &b_L1_SingleEG8_00001); 
  fChain->SetBranchAddress("L1_SingleForJet10", &map_BitOfStandardHLTPath["L1_SingleForJet10"], &b_L1_SingleForJet10); 
  fChain->SetBranchAddress("L1_SingleForJet6", &map_BitOfStandardHLTPath["L1_SingleForJet6"], &b_L1_SingleForJet6); 
  fChain->SetBranchAddress("L1_SingleHfBitCountsRing1_1", &map_BitOfStandardHLTPath["L1_SingleHfBitCountsRing1_1"], &b_L1_SingleHfBitCountsRing1_1); 
  fChain->SetBranchAddress("L1_SingleHfBitCountsRing2_1", &map_BitOfStandardHLTPath["L1_SingleHfBitCountsRing2_1"], &b_L1_SingleHfBitCountsRing2_1); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing1_200", &map_BitOfStandardHLTPath["L1_SingleHfRingEtSumsRing1_200"], &b_L1_SingleHfRingEtSumsRing1_200); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing1_4", &map_BitOfStandardHLTPath["L1_SingleHfRingEtSumsRing1_4"], &b_L1_SingleHfRingEtSumsRing1_4); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing2_200", &map_BitOfStandardHLTPath["L1_SingleHfRingEtSumsRing2_200"], &b_L1_SingleHfRingEtSumsRing2_200); 
  fChain->SetBranchAddress("L1_SingleHfRingEtSumsRing2_4", &map_BitOfStandardHLTPath["L1_SingleHfRingEtSumsRing2_4"], &b_L1_SingleHfRingEtSumsRing2_4); 
  fChain->SetBranchAddress("L1_SingleIsoEG10_00001", &map_BitOfStandardHLTPath["L1_SingleIsoEG10_00001"], &b_L1_SingleIsoEG10_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG12_00001", &map_BitOfStandardHLTPath["L1_SingleIsoEG12_00001"], &b_L1_SingleIsoEG12_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG15_00001", &map_BitOfStandardHLTPath["L1_SingleIsoEG15_00001"], &b_L1_SingleIsoEG15_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG5_00001", &map_BitOfStandardHLTPath["L1_SingleIsoEG5_00001"], &b_L1_SingleIsoEG5_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG5_Endcap_00001", &map_BitOfStandardHLTPath["L1_SingleIsoEG5_Endcap_00001"], &b_L1_SingleIsoEG5_Endcap_00001); 
  fChain->SetBranchAddress("L1_SingleIsoEG8_00001", &map_BitOfStandardHLTPath["L1_SingleIsoEG8_00001"], &b_L1_SingleIsoEG8_00001); 
  fChain->SetBranchAddress("L1_SingleJet10_00001", &map_BitOfStandardHLTPath["L1_SingleJet10_00001"], &b_L1_SingleJet10_00001); 
  fChain->SetBranchAddress("L1_SingleJet10_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleJet10_Barrel_00001"], &b_L1_SingleJet10_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet10_Central", &map_BitOfStandardHLTPath["L1_SingleJet10_Central"], &b_L1_SingleJet10_Central); 
  fChain->SetBranchAddress("L1_SingleJet10_Endcap", &map_BitOfStandardHLTPath["L1_SingleJet10_Endcap"], &b_L1_SingleJet10_Endcap); 
  fChain->SetBranchAddress("L1_SingleJet20_00001", &map_BitOfStandardHLTPath["L1_SingleJet20_00001"], &b_L1_SingleJet20_00001); 
  fChain->SetBranchAddress("L1_SingleJet20_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleJet20_Barrel_00001"], &b_L1_SingleJet20_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet30_00001", &map_BitOfStandardHLTPath["L1_SingleJet30_00001"], &b_L1_SingleJet30_00001); 
  fChain->SetBranchAddress("L1_SingleJet30_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleJet30_Barrel_00001"], &b_L1_SingleJet30_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet40_00001", &map_BitOfStandardHLTPath["L1_SingleJet40_00001"], &b_L1_SingleJet40_00001); 
  fChain->SetBranchAddress("L1_SingleJet40_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleJet40_Barrel_00001"], &b_L1_SingleJet40_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet50_00001", &map_BitOfStandardHLTPath["L1_SingleJet50_00001"], &b_L1_SingleJet50_00001); 
  fChain->SetBranchAddress("L1_SingleJet60_00001", &map_BitOfStandardHLTPath["L1_SingleJet60_00001"], &b_L1_SingleJet60_00001); 
  fChain->SetBranchAddress("L1_SingleJet6_00001", &map_BitOfStandardHLTPath["L1_SingleJet6_00001"], &b_L1_SingleJet6_00001); 
  fChain->SetBranchAddress("L1_SingleJet6_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleJet6_Barrel_00001"], &b_L1_SingleJet6_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleJet6_Central", &map_BitOfStandardHLTPath["L1_SingleJet6_Central"], &b_L1_SingleJet6_Central); 
  fChain->SetBranchAddress("L1_SingleJet6_Endcap", &map_BitOfStandardHLTPath["L1_SingleJet6_Endcap"], &b_L1_SingleJet6_Endcap); 
  fChain->SetBranchAddress("L1_SingleMu0", &map_BitOfStandardHLTPath["L1_SingleMu0"], &b_L1_SingleMu0); 
  fChain->SetBranchAddress("L1_SingleMu10", &map_BitOfStandardHLTPath["L1_SingleMu10"], &b_L1_SingleMu10); 
  fChain->SetBranchAddress("L1_SingleMu14", &map_BitOfStandardHLTPath["L1_SingleMu14"], &b_L1_SingleMu14); 
  fChain->SetBranchAddress("L1_SingleMu3", &map_BitOfStandardHLTPath["L1_SingleMu3"], &b_L1_SingleMu3); 
  fChain->SetBranchAddress("L1_SingleMu5", &map_BitOfStandardHLTPath["L1_SingleMu5"], &b_L1_SingleMu5); 
  fChain->SetBranchAddress("L1_SingleMu7", &map_BitOfStandardHLTPath["L1_SingleMu7"], &b_L1_SingleMu7); 
  fChain->SetBranchAddress("L1_SingleMuBeamHalo", &map_BitOfStandardHLTPath["L1_SingleMuBeamHalo"], &b_L1_SingleMuBeamHalo); 
  fChain->SetBranchAddress("L1_SingleMuOpen", &map_BitOfStandardHLTPath["L1_SingleMuOpen"], &b_L1_SingleMuOpen); 
  fChain->SetBranchAddress("L1_SingleTauJet10_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet10_00001"], &b_L1_SingleTauJet10_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet10_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet10_Barrel_00001"], &b_L1_SingleTauJet10_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet20_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet20_00001"], &b_L1_SingleTauJet20_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet20_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet20_Barrel_00001"], &b_L1_SingleTauJet20_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet30_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet30_00001"], &b_L1_SingleTauJet30_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet30_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet30_Barrel_00001"], &b_L1_SingleTauJet30_Barrel_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet50_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet50_00001"], &b_L1_SingleTauJet50_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet8_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet8_00001"], &b_L1_SingleTauJet8_00001); 
  fChain->SetBranchAddress("L1_SingleTauJet8_Barrel_00001", &map_BitOfStandardHLTPath["L1_SingleTauJet8_Barrel_00001"], &b_L1_SingleTauJet8_Barrel_00001); 
  fChain->SetBranchAddress("L1_TauJet10_ETM30_00001", &map_BitOfStandardHLTPath["L1_TauJet10_ETM30_00001"], &b_L1_TauJet10_ETM30_00001); 
  fChain->SetBranchAddress("L1_TauJet10_ETM40_00001", &map_BitOfStandardHLTPath["L1_TauJet10_ETM40_00001"], &b_L1_TauJet10_ETM40_00001); 
  fChain->SetBranchAddress("L1_TripleJet30_00001", &map_BitOfStandardHLTPath["L1_TripleJet30_00001"], &b_L1_TripleJet30_00001); 
  fChain->SetBranchAddress("L1_TripleMu3", &map_BitOfStandardHLTPath["L1_TripleMu3"], &b_L1_TripleMu3); 

  fChain->SetBranchAddress("HLT_L1Jet15", &map_BitOfStandardHLTPath["HLT_L1Jet15"], &b_HLT_L1Jet15);
  fChain->SetBranchAddress("HLT_Jet30", &map_BitOfStandardHLTPath["HLT_Jet30"], &b_HLT_Jet30);
  fChain->SetBranchAddress("HLT_Jet50", &map_BitOfStandardHLTPath["HLT_Jet50"], &b_HLT_Jet50);
  fChain->SetBranchAddress("HLT_Jet80", &map_BitOfStandardHLTPath["HLT_Jet80"], &b_HLT_Jet80);
  fChain->SetBranchAddress("HLT_Jet110", &map_BitOfStandardHLTPath["HLT_Jet110"], &b_HLT_Jet110);
  fChain->SetBranchAddress("HLT_Jet180", &map_BitOfStandardHLTPath["HLT_Jet180"], &b_HLT_Jet180);
  fChain->SetBranchAddress("HLT_Jet250", &map_BitOfStandardHLTPath["HLT_Jet250"], &b_HLT_Jet250);
  fChain->SetBranchAddress("HLT_FwdJet20", &map_BitOfStandardHLTPath["HLT_FwdJet20"], &b_HLT_FwdJet20);
  fChain->SetBranchAddress("HLT_DoubleJet150", &map_BitOfStandardHLTPath["HLT_DoubleJet150"], &b_HLT_DoubleJet150);
  fChain->SetBranchAddress("HLT_DoubleJet125_Aco", &map_BitOfStandardHLTPath["HLT_DoubleJet125_Aco"], &b_HLT_DoubleJet125_Aco);
  fChain->SetBranchAddress("HLT_DoubleFwdJet50", &map_BitOfStandardHLTPath["HLT_DoubleFwdJet50"], &b_HLT_DoubleFwdJet50);
  fChain->SetBranchAddress("HLT_DiJetAve15", &map_BitOfStandardHLTPath["HLT_DiJetAve15"], &b_HLT_DiJetAve15);
  fChain->SetBranchAddress("HLT_DiJetAve30", &map_BitOfStandardHLTPath["HLT_DiJetAve30"], &b_HLT_DiJetAve30);
  fChain->SetBranchAddress("HLT_DiJetAve50", &map_BitOfStandardHLTPath["HLT_DiJetAve50"], &b_HLT_DiJetAve50);
  fChain->SetBranchAddress("HLT_DiJetAve70", &map_BitOfStandardHLTPath["HLT_DiJetAve70"], &b_HLT_DiJetAve70);
  fChain->SetBranchAddress("HLT_DiJetAve130", &map_BitOfStandardHLTPath["HLT_DiJetAve130"], &b_HLT_DiJetAve130);
  fChain->SetBranchAddress("HLT_DiJetAve220", &map_BitOfStandardHLTPath["HLT_DiJetAve220"], &b_HLT_DiJetAve220);
  fChain->SetBranchAddress("HLT_TripleJet85", &map_BitOfStandardHLTPath["HLT_TripleJet85"], &b_HLT_TripleJet85);
  fChain->SetBranchAddress("HLT_QuadJet30", &map_BitOfStandardHLTPath["HLT_QuadJet30"], &b_HLT_QuadJet30);
  fChain->SetBranchAddress("HLT_QuadJet60", &map_BitOfStandardHLTPath["HLT_QuadJet60"], &b_HLT_QuadJet60);
  fChain->SetBranchAddress("HLT_SumET120", &map_BitOfStandardHLTPath["HLT_SumET120"], &b_HLT_SumET120);
  fChain->SetBranchAddress("HLT_L1MET20", &map_BitOfStandardHLTPath["HLT_L1MET20"], &b_HLT_L1MET20);
  fChain->SetBranchAddress("HLT_MET25", &map_BitOfStandardHLTPath["HLT_MET25"], &b_HLT_MET25);
  fChain->SetBranchAddress("HLT_MET35", &map_BitOfStandardHLTPath["HLT_MET35"], &b_HLT_MET35);
  fChain->SetBranchAddress("HLT_MET50", &map_BitOfStandardHLTPath["HLT_MET50"], &b_HLT_MET50);
  fChain->SetBranchAddress("HLT_MET65", &map_BitOfStandardHLTPath["HLT_MET65"], &b_HLT_MET65);
  fChain->SetBranchAddress("HLT_MET75", &map_BitOfStandardHLTPath["HLT_MET75"], &b_HLT_MET75);
  fChain->SetBranchAddress("HLT_MET35_HT350", &map_BitOfStandardHLTPath["HLT_MET35_HT350"], &b_HLT_MET35_HT350);
  fChain->SetBranchAddress("HLT_Jet180_MET60", &map_BitOfStandardHLTPath["HLT_Jet180_MET60"], &b_HLT_Jet180_MET60);
  fChain->SetBranchAddress("HLT_Jet60_MET70_Aco", &map_BitOfStandardHLTPath["HLT_Jet60_MET70_Aco"], &b_HLT_Jet60_MET70_Aco);
  fChain->SetBranchAddress("HLT_Jet100_MET60_Aco", &map_BitOfStandardHLTPath["HLT_Jet100_MET60_Aco"], &b_HLT_Jet100_MET60_Aco);
  fChain->SetBranchAddress("HLT_DoubleJet125_MET60", &map_BitOfStandardHLTPath["HLT_DoubleJet125_MET60"], &b_HLT_DoubleJet125_MET60);
  fChain->SetBranchAddress("HLT_DoubleFwdJet40_MET60", &map_BitOfStandardHLTPath["HLT_DoubleFwdJet40_MET60"], &b_HLT_DoubleFwdJet40_MET60);
  fChain->SetBranchAddress("HLT_DoubleJet60_MET60_Aco", &map_BitOfStandardHLTPath["HLT_DoubleJet60_MET60_Aco"], &b_HLT_DoubleJet60_MET60_Aco);
  fChain->SetBranchAddress("HLT_DoubleJet50_MET70_Aco", &map_BitOfStandardHLTPath["HLT_DoubleJet50_MET70_Aco"], &b_HLT_DoubleJet50_MET70_Aco);
  fChain->SetBranchAddress("HLT_DoubleJet40_MET70_Aco", &map_BitOfStandardHLTPath["HLT_DoubleJet40_MET70_Aco"], &b_HLT_DoubleJet40_MET70_Aco);
  fChain->SetBranchAddress("HLT_TripleJet60_MET60", &map_BitOfStandardHLTPath["HLT_TripleJet60_MET60"], &b_HLT_TripleJet60_MET60);
  fChain->SetBranchAddress("HLT_QuadJet35_MET60", &map_BitOfStandardHLTPath["HLT_QuadJet35_MET60"], &b_HLT_QuadJet35_MET60);
  fChain->SetBranchAddress("HLT_IsoEle15_L1I", &map_BitOfStandardHLTPath["HLT_IsoEle15_L1I"], &b_HLT_IsoEle15_L1I);
  fChain->SetBranchAddress("HLT_IsoEle18_L1R", &map_BitOfStandardHLTPath["HLT_IsoEle18_L1R"], &b_HLT_IsoEle18_L1R);
  fChain->SetBranchAddress("HLT_IsoEle15_LW_L1I", &map_BitOfStandardHLTPath["HLT_IsoEle15_LW_L1I"], &b_HLT_IsoEle15_LW_L1I);
  fChain->SetBranchAddress("HLT_LooseIsoEle15_LW_L1R", &map_BitOfStandardHLTPath["HLT_LooseIsoEle15_LW_L1R"], &b_HLT_LooseIsoEle15_LW_L1R);
  fChain->SetBranchAddress("HLT_Ele10_SW_L1R", &map_BitOfStandardHLTPath["HLT_Ele10_SW_L1R"], &b_HLT_Ele10_SW_L1R);
  fChain->SetBranchAddress("HLT_Ele15_SW_L1R", &map_BitOfStandardHLTPath["HLT_Ele15_SW_L1R"], &b_HLT_Ele15_SW_L1R);
  fChain->SetBranchAddress("HLT_Ele15_LW_L1R", &map_BitOfStandardHLTPath["HLT_Ele15_LW_L1R"], &b_HLT_Ele15_LW_L1R);
  fChain->SetBranchAddress("HLT_EM80", &map_BitOfStandardHLTPath["HLT_EM80"], &b_HLT_EM80);
  fChain->SetBranchAddress("HLT_EM200", &map_BitOfStandardHLTPath["HLT_EM200"], &b_HLT_EM200);
  fChain->SetBranchAddress("HLT_DoubleIsoEle10_L1I", &map_BitOfStandardHLTPath["HLT_DoubleIsoEle10_L1I"], &b_HLT_DoubleIsoEle10_L1I);
  fChain->SetBranchAddress("HLT_DoubleIsoEle12_L1R", &map_BitOfStandardHLTPath["HLT_DoubleIsoEle12_L1R"], &b_HLT_DoubleIsoEle12_L1R);
  fChain->SetBranchAddress("HLT_DoubleIsoEle10_LW_L1I", &map_BitOfStandardHLTPath["HLT_DoubleIsoEle10_LW_L1I"], &b_HLT_DoubleIsoEle10_LW_L1I);
  fChain->SetBranchAddress("HLT_DoubleIsoEle12_LW_L1R", &map_BitOfStandardHLTPath["HLT_DoubleIsoEle12_LW_L1R"], &b_HLT_DoubleIsoEle12_LW_L1R);
  fChain->SetBranchAddress("HLT_DoubleEle5_SW_L1R", &map_BitOfStandardHLTPath["HLT_DoubleEle5_SW_L1R"], &b_HLT_DoubleEle5_SW_L1R);
  fChain->SetBranchAddress("HLT_DoubleEle10_LW_OnlyPixelM_L1R", &map_BitOfStandardHLTPath["HLT_DoubleEle10_LW_OnlyPixelM_L1R"], &b_HLT_DoubleEle10_LW_OnlyPixelM_L1R);
  fChain->SetBranchAddress("HLT_DoubleEle10_Z", &map_BitOfStandardHLTPath["HLT_DoubleEle10_Z"], &b_HLT_DoubleEle10_Z);
  fChain->SetBranchAddress("HLT_DoubleEle6_Exclusive", &map_BitOfStandardHLTPath["HLT_DoubleEle6_Exclusive"], &b_HLT_DoubleEle6_Exclusive);
  fChain->SetBranchAddress("HLT_IsoPhoton30_L1I", &map_BitOfStandardHLTPath["HLT_IsoPhoton30_L1I"], &b_HLT_IsoPhoton30_L1I);
  fChain->SetBranchAddress("HLT_IsoPhoton10_L1R", &map_BitOfStandardHLTPath["HLT_IsoPhoton10_L1R"], &b_HLT_IsoPhoton10_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton15_L1R", &map_BitOfStandardHLTPath["HLT_IsoPhoton15_L1R"], &b_HLT_IsoPhoton15_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton20_L1R", &map_BitOfStandardHLTPath["HLT_IsoPhoton20_L1R"], &b_HLT_IsoPhoton20_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton25_L1R", &map_BitOfStandardHLTPath["HLT_IsoPhoton25_L1R"], &b_HLT_IsoPhoton25_L1R);
  fChain->SetBranchAddress("HLT_IsoPhoton40_L1R", &map_BitOfStandardHLTPath["HLT_IsoPhoton40_L1R"], &b_HLT_IsoPhoton40_L1R);
  fChain->SetBranchAddress("HLT_Photon15_L1R", &map_BitOfStandardHLTPath["HLT_Photon15_L1R"], &b_HLT_Photon15_L1R);
  fChain->SetBranchAddress("HLT_Photon25_L1R", &map_BitOfStandardHLTPath["HLT_Photon25_L1R"], &b_HLT_Photon25_L1R);
  fChain->SetBranchAddress("HLT_DoubleIsoPhoton20_L1I", &map_BitOfStandardHLTPath["HLT_DoubleIsoPhoton20_L1I"], &b_HLT_DoubleIsoPhoton20_L1I);
  fChain->SetBranchAddress("HLT_DoubleIsoPhoton20_L1R", &map_BitOfStandardHLTPath["HLT_DoubleIsoPhoton20_L1R"], &b_HLT_DoubleIsoPhoton20_L1R);
  fChain->SetBranchAddress("HLT_DoublePhoton10_Exclusive", &map_BitOfStandardHLTPath["HLT_DoublePhoton10_Exclusive"], &b_HLT_DoublePhoton10_Exclusive);
  fChain->SetBranchAddress("HLT_L1Mu", &map_BitOfStandardHLTPath["HLT_L1Mu"], &b_HLT_L1Mu);
  fChain->SetBranchAddress("HLT_L1MuOpen", &map_BitOfStandardHLTPath["HLT_L1MuOpen"], &b_HLT_L1MuOpen);
  fChain->SetBranchAddress("HLT_L2Mu9", &map_BitOfStandardHLTPath["HLT_L2Mu9"], &b_HLT_L2Mu9);
  fChain->SetBranchAddress("HLT_IsoMu9", &map_BitOfStandardHLTPath["HLT_IsoMu9"], &b_HLT_IsoMu9);
  fChain->SetBranchAddress("HLT_IsoMu11", &map_BitOfStandardHLTPath["HLT_IsoMu11"], &b_HLT_IsoMu11);
  fChain->SetBranchAddress("HLT_IsoMu13", &map_BitOfStandardHLTPath["HLT_IsoMu13"], &b_HLT_IsoMu13);
  fChain->SetBranchAddress("HLT_IsoMu15", &map_BitOfStandardHLTPath["HLT_IsoMu15"], &b_HLT_IsoMu15);
  fChain->SetBranchAddress("HLT_NoTrackerIsoMu15", &map_BitOfStandardHLTPath["HLT_NoTrackerIsoMu15"], &b_HLT_NoTrackerIsoMu15);
  fChain->SetBranchAddress("HLT_Mu3", &map_BitOfStandardHLTPath["HLT_Mu3"], &b_HLT_Mu3);
  fChain->SetBranchAddress("HLT_Mu5", &map_BitOfStandardHLTPath["HLT_Mu5"], &b_HLT_Mu5);
  fChain->SetBranchAddress("HLT_Mu7", &map_BitOfStandardHLTPath["HLT_Mu7"], &b_HLT_Mu7);
  fChain->SetBranchAddress("HLT_Mu9", &map_BitOfStandardHLTPath["HLT_Mu9"], &b_HLT_Mu9);
  fChain->SetBranchAddress("HLT_Mu11", &map_BitOfStandardHLTPath["HLT_Mu11"], &b_HLT_Mu11);
  fChain->SetBranchAddress("HLT_Mu13", &map_BitOfStandardHLTPath["HLT_Mu13"], &b_HLT_Mu13);
  fChain->SetBranchAddress("HLT_Mu15", &map_BitOfStandardHLTPath["HLT_Mu15"], &b_HLT_Mu15);
  fChain->SetBranchAddress("HLT_Mu15_L1Mu7", &map_BitOfStandardHLTPath["HLT_Mu15_L1Mu7"], &b_HLT_Mu15_L1Mu7);
  fChain->SetBranchAddress("HLT_Mu15_Vtx2cm", &map_BitOfStandardHLTPath["HLT_Mu15_Vtx2cm"], &b_HLT_Mu15_Vtx2cm);
  fChain->SetBranchAddress("HLT_Mu15_Vtx2mm", &map_BitOfStandardHLTPath["HLT_Mu15_Vtx2mm"], &b_HLT_Mu15_Vtx2mm);
  fChain->SetBranchAddress("HLT_DoubleIsoMu3", &map_BitOfStandardHLTPath["HLT_DoubleIsoMu3"], &b_HLT_DoubleIsoMu3);
  fChain->SetBranchAddress("HLT_DoubleMu3", &map_BitOfStandardHLTPath["HLT_DoubleMu3"], &b_HLT_DoubleMu3);
  fChain->SetBranchAddress("HLT_DoubleMu3_Vtx2cm", &map_BitOfStandardHLTPath["HLT_DoubleMu3_Vtx2cm"], &b_HLT_DoubleMu3_Vtx2cm);
  fChain->SetBranchAddress("HLT_DoubleMu3_Vtx2mm", &map_BitOfStandardHLTPath["HLT_DoubleMu3_Vtx2mm"], &b_HLT_DoubleMu3_Vtx2mm);
  fChain->SetBranchAddress("HLT_DoubleMu3_JPsi", &map_BitOfStandardHLTPath["HLT_DoubleMu3_JPsi"], &b_HLT_DoubleMu3_JPsi);
  fChain->SetBranchAddress("HLT_DoubleMu3_Upsilon", &map_BitOfStandardHLTPath["HLT_DoubleMu3_Upsilon"], &b_HLT_DoubleMu3_Upsilon);
  fChain->SetBranchAddress("HLT_DoubleMu7_Z", &map_BitOfStandardHLTPath["HLT_DoubleMu7_Z"], &b_HLT_DoubleMu7_Z);
  fChain->SetBranchAddress("HLT_DoubleMu3_SameSign", &map_BitOfStandardHLTPath["HLT_DoubleMu3_SameSign"], &b_HLT_DoubleMu3_SameSign);
  fChain->SetBranchAddress("HLT_DoubleMu3_Psi2S", &map_BitOfStandardHLTPath["HLT_DoubleMu3_Psi2S"], &b_HLT_DoubleMu3_Psi2S);
  fChain->SetBranchAddress("HLT_BTagIP_Jet180", &map_BitOfStandardHLTPath["HLT_BTagIP_Jet180"], &b_HLT_BTagIP_Jet180);
  fChain->SetBranchAddress("HLT_BTagIP_Jet120_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagIP_Jet120_Relaxed"], &b_HLT_BTagIP_Jet120_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_DoubleJet120", &map_BitOfStandardHLTPath["HLT_BTagIP_DoubleJet120"], &b_HLT_BTagIP_DoubleJet120);
  fChain->SetBranchAddress("HLT_BTagIP_DoubleJet60_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagIP_DoubleJet60_Relaxed"], &b_HLT_BTagIP_DoubleJet60_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_TripleJet70", &map_BitOfStandardHLTPath["HLT_BTagIP_TripleJet70"], &b_HLT_BTagIP_TripleJet70);
  fChain->SetBranchAddress("HLT_BTagIP_TripleJet40_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagIP_TripleJet40_Relaxed"], &b_HLT_BTagIP_TripleJet40_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_QuadJet40", &map_BitOfStandardHLTPath["HLT_BTagIP_QuadJet40"], &b_HLT_BTagIP_QuadJet40);
  fChain->SetBranchAddress("HLT_BTagIP_QuadJet30_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagIP_QuadJet30_Relaxed"], &b_HLT_BTagIP_QuadJet30_Relaxed);
  fChain->SetBranchAddress("HLT_BTagIP_HT470", &map_BitOfStandardHLTPath["HLT_BTagIP_HT470"], &b_HLT_BTagIP_HT470);
  fChain->SetBranchAddress("HLT_BTagIP_HT320_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagIP_HT320_Relaxed"], &b_HLT_BTagIP_HT320_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_DoubleJet120", &map_BitOfStandardHLTPath["HLT_BTagMu_DoubleJet120"], &b_HLT_BTagMu_DoubleJet120);
  fChain->SetBranchAddress("HLT_BTagMu_DoubleJet60_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagMu_DoubleJet60_Relaxed"], &b_HLT_BTagMu_DoubleJet60_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_TripleJet70", &map_BitOfStandardHLTPath["HLT_BTagMu_TripleJet70"], &b_HLT_BTagMu_TripleJet70);
  fChain->SetBranchAddress("HLT_BTagMu_TripleJet40_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagMu_TripleJet40_Relaxed"], &b_HLT_BTagMu_TripleJet40_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_QuadJet40", &map_BitOfStandardHLTPath["HLT_BTagMu_QuadJet40"], &b_HLT_BTagMu_QuadJet40);
  fChain->SetBranchAddress("HLT_BTagMu_QuadJet30_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagMu_QuadJet30_Relaxed"], &b_HLT_BTagMu_QuadJet30_Relaxed);
  fChain->SetBranchAddress("HLT_BTagMu_HT370", &map_BitOfStandardHLTPath["HLT_BTagMu_HT370"], &b_HLT_BTagMu_HT370);
  fChain->SetBranchAddress("HLT_BTagMu_HT250_Relaxed", &map_BitOfStandardHLTPath["HLT_BTagMu_HT250_Relaxed"], &b_HLT_BTagMu_HT250_Relaxed);
  fChain->SetBranchAddress("HLT_DoubleMu3_BJPsi", &map_BitOfStandardHLTPath["HLT_DoubleMu3_BJPsi"], &b_HLT_DoubleMu3_BJPsi);
  fChain->SetBranchAddress("HLT_DoubleMu4_BJPsi", &map_BitOfStandardHLTPath["HLT_DoubleMu4_BJPsi"], &b_HLT_DoubleMu4_BJPsi);
  fChain->SetBranchAddress("HLT_TripleMu3_TauTo3Mu", &map_BitOfStandardHLTPath["HLT_TripleMu3_TauTo3Mu"], &b_HLT_TripleMu3_TauTo3Mu);
  fChain->SetBranchAddress("HLT_IsoTau_MET65_Trk20", &map_BitOfStandardHLTPath["HLT_IsoTau_MET65_Trk20"], &b_HLT_IsoTau_MET65_Trk20);
  fChain->SetBranchAddress("HLT_IsoTau_MET35_Trk15_L1MET", &map_BitOfStandardHLTPath["HLT_IsoTau_MET35_Trk15_L1MET"], &b_HLT_IsoTau_MET35_Trk15_L1MET);
  fChain->SetBranchAddress("HLT_LooseIsoTau_MET30", &map_BitOfStandardHLTPath["HLT_LooseIsoTau_MET30"], &b_HLT_LooseIsoTau_MET30);
  fChain->SetBranchAddress("HLT_LooseIsoTau_MET30_L1MET", &map_BitOfStandardHLTPath["HLT_LooseIsoTau_MET30_L1MET"], &b_HLT_LooseIsoTau_MET30_L1MET);
  fChain->SetBranchAddress("HLT_DoubleIsoTau_Trk3", &map_BitOfStandardHLTPath["HLT_DoubleIsoTau_Trk3"], &b_HLT_DoubleIsoTau_Trk3);
  fChain->SetBranchAddress("HLT_DoubleLooseIsoTau", &map_BitOfStandardHLTPath["HLT_DoubleLooseIsoTau"], &b_HLT_DoubleLooseIsoTau);
  fChain->SetBranchAddress("HLT_IsoEle8_IsoMu7", &map_BitOfStandardHLTPath["HLT_IsoEle8_IsoMu7"], &b_HLT_IsoEle8_IsoMu7);
  fChain->SetBranchAddress("HLT_IsoEle10_Mu10_L1R", &map_BitOfStandardHLTPath["HLT_IsoEle10_Mu10_L1R"], &b_HLT_IsoEle10_Mu10_L1R);
  fChain->SetBranchAddress("HLT_IsoEle12_IsoTau_Trk3", &map_BitOfStandardHLTPath["HLT_IsoEle12_IsoTau_Trk3"], &b_HLT_IsoEle12_IsoTau_Trk3);
  fChain->SetBranchAddress("HLT_IsoEle10_BTagIP_Jet35", &map_BitOfStandardHLTPath["HLT_IsoEle10_BTagIP_Jet35"], &b_HLT_IsoEle10_BTagIP_Jet35);
  fChain->SetBranchAddress("HLT_IsoEle12_Jet40", &map_BitOfStandardHLTPath["HLT_IsoEle12_Jet40"], &b_HLT_IsoEle12_Jet40);
  fChain->SetBranchAddress("HLT_IsoEle12_DoubleJet80", &map_BitOfStandardHLTPath["HLT_IsoEle12_DoubleJet80"], &b_HLT_IsoEle12_DoubleJet80);
  fChain->SetBranchAddress("HLT_IsoElec5_TripleJet30", &map_BitOfStandardHLTPath["HLT_IsoElec5_TripleJet30"], &b_HLT_IsoElec5_TripleJet30);
  fChain->SetBranchAddress("HLT_IsoEle12_TripleJet60", &map_BitOfStandardHLTPath["HLT_IsoEle12_TripleJet60"], &b_HLT_IsoEle12_TripleJet60);
  fChain->SetBranchAddress("HLT_IsoEle12_QuadJet35", &map_BitOfStandardHLTPath["HLT_IsoEle12_QuadJet35"], &b_HLT_IsoEle12_QuadJet35);
  fChain->SetBranchAddress("HLT_IsoMu14_IsoTau_Trk3", &map_BitOfStandardHLTPath["HLT_IsoMu14_IsoTau_Trk3"], &b_HLT_IsoMu14_IsoTau_Trk3);
  fChain->SetBranchAddress("HLT_IsoMu7_BTagIP_Jet35", &map_BitOfStandardHLTPath["HLT_IsoMu7_BTagIP_Jet35"], &b_HLT_IsoMu7_BTagIP_Jet35);
  fChain->SetBranchAddress("HLT_IsoMu7_BTagMu_Jet20", &map_BitOfStandardHLTPath["HLT_IsoMu7_BTagMu_Jet20"], &b_HLT_IsoMu7_BTagMu_Jet20);
  fChain->SetBranchAddress("HLT_IsoMu7_Jet40", &map_BitOfStandardHLTPath["HLT_IsoMu7_Jet40"], &b_HLT_IsoMu7_Jet40);
  fChain->SetBranchAddress("HLT_NoL2IsoMu8_Jet40", &map_BitOfStandardHLTPath["HLT_NoL2IsoMu8_Jet40"], &b_HLT_NoL2IsoMu8_Jet40);
  fChain->SetBranchAddress("HLT_Mu14_Jet50", &map_BitOfStandardHLTPath["HLT_Mu14_Jet50"], &b_HLT_Mu14_Jet50);
  fChain->SetBranchAddress("HLT_Mu5_TripleJet30", &map_BitOfStandardHLTPath["HLT_Mu5_TripleJet30"], &b_HLT_Mu5_TripleJet30);
  fChain->SetBranchAddress("HLT_BTagMu_Jet20_Calib", &map_BitOfStandardHLTPath["HLT_BTagMu_Jet20_Calib"], &b_HLT_BTagMu_Jet20_Calib);
  fChain->SetBranchAddress("HLT_ZeroBias", &map_BitOfStandardHLTPath["HLT_ZeroBias"], &b_HLT_ZeroBias);
  fChain->SetBranchAddress("HLT_MinBias", &map_BitOfStandardHLTPath["HLT_MinBias"], &b_HLT_MinBias);
  fChain->SetBranchAddress("HLT_MinBiasHcal", &map_BitOfStandardHLTPath["HLT_MinBiasHcal"], &b_HLT_MinBiasHcal);
  fChain->SetBranchAddress("HLT_MinBiasEcal", &map_BitOfStandardHLTPath["HLT_MinBiasEcal"], &b_HLT_MinBiasEcal);
  fChain->SetBranchAddress("HLT_MinBiasPixel", &map_BitOfStandardHLTPath["HLT_MinBiasPixel"], &b_HLT_MinBiasPixel);
  fChain->SetBranchAddress("HLT_MinBiasPixel_Trk5", &map_BitOfStandardHLTPath["HLT_MinBiasPixel_Trk5"], &b_HLT_MinBiasPixel_Trk5);
  fChain->SetBranchAddress("HLT_BackwardBSC", &map_BitOfStandardHLTPath["HLT_BackwardBSC"], &b_HLT_BackwardBSC);
  fChain->SetBranchAddress("HLT_ForwardBSC", &map_BitOfStandardHLTPath["HLT_ForwardBSC"], &b_HLT_ForwardBSC);
  fChain->SetBranchAddress("HLT_CSCBeamHalo", &map_BitOfStandardHLTPath["HLT_CSCBeamHalo"], &b_HLT_CSCBeamHalo);
  fChain->SetBranchAddress("HLT_CSCBeamHaloOverlapRing1", &map_BitOfStandardHLTPath["HLT_CSCBeamHaloOverlapRing1"], &b_HLT_CSCBeamHaloOverlapRing1);
  fChain->SetBranchAddress("HLT_CSCBeamHaloOverlapRing2", &map_BitOfStandardHLTPath["HLT_CSCBeamHaloOverlapRing2"], &b_HLT_CSCBeamHaloOverlapRing2);
  fChain->SetBranchAddress("HLT_CSCBeamHaloRing2or3", &map_BitOfStandardHLTPath["HLT_CSCBeamHaloRing2or3"], &b_HLT_CSCBeamHaloRing2or3);
  fChain->SetBranchAddress("HLT_TrackerCosmics", &map_BitOfStandardHLTPath["HLT_TrackerCosmics"], &b_HLT_TrackerCosmics);
  fChain->SetBranchAddress("HLT_TriggerType", &map_BitOfStandardHLTPath["HLT_TriggerType"], &b_HLT_TriggerType);
  fChain->SetBranchAddress("AlCa_IsoTrack", &map_BitOfStandardHLTPath["AlCa_IsoTrack"], &b_AlCa_IsoTrack);
  fChain->SetBranchAddress("AlCa_EcalPhiSym", &map_BitOfStandardHLTPath["AlCa_EcalPhiSym"], &b_AlCa_EcalPhiSym);
  fChain->SetBranchAddress("AlCa_EcalPi0", &map_BitOfStandardHLTPath["AlCa_EcalPi0"], &b_AlCa_EcalPi0);

  
  Notify();
}

void OHltTree::SetMapL1SeedsOfStandardHLTPath(OHltMenu *menu) {
  map_L1SeedsOfStandardHLTPath = menu->GetL1SeedsOfHLTPathMap();
}

void OHltTree::ApplyL1Prescales(OHltMenu *menu)
{
  TString st;
  unsigned int tt = menu->GetL1TriggerSize();
  for (unsigned int i=0;i<tt;i++) {
    st = menu->GetL1TriggerName(i);
    if (map_BitOfStandardHLTPath.find(st)->second == 1) {
      if (GetIntRandom() % menu->GetL1Prescale(i) != 0) {
	map_BitOfStandardHLTPath[st] = 0;	
      }
    }
  }
}

void OHltTree::SetMapL1BitOfStandardHLTPath(OHltMenu *menu) {
  int tt = 0;
  TString st;
  unsigned ts = menu->GetTriggerSize();
  for (unsigned int i=0;i<ts;i++) {
    st = menu->GetTriggerName(i);
    map< TString, vector<TString> >::const_iterator it = map_L1SeedsOfStandardHLTPath.find(st);
    if (it != map_L1SeedsOfStandardHLTPath.end()) {
      tt = 0;
      unsigned ts2 = it->second.size();
      for (unsigned int j=0;j<ts2;j++) {
 	tt += (map_BitOfStandardHLTPath.find((map_L1SeedsOfStandardHLTPath.find(st)->second)[j]))->second;
      }
    }
    map_L1BitOfStandardHLTPath[st] = tt;
  }
}

void OHltTree::SetL1MuonQuality()
{
  // Cut on muon quality
  // init
  for (int i=0;i<10;i++) {
    NL1OpenMu = 0;
    L1OpenMuPt[i] = -999.;
    L1OpenMuE[i] = -999.;
    L1OpenMuEta[i] = -999.;
    L1OpenMuPhi[i] = -999.;
    L1OpenMuIsol[i] = -999;
    L1OpenMuMip[i] = -999;
    L1OpenMuFor[i] = -999;
    L1OpenMuRPC[i] = -999;
    L1OpenMuQal[i] = -999;     
  }
  for (int i=0;i<NL1Mu;i++) {
    if ( L1MuQal[i]==2 || L1MuQal[i]==3 || L1MuQal[i]==4 ||
	 L1MuQal[i]==5 || L1MuQal[i]==6 || L1MuQal[i]==7 ) {
      L1OpenMuPt[NL1OpenMu] = L1MuPt[i];
      L1OpenMuE[NL1OpenMu] = L1MuE[i];
      L1OpenMuEta[NL1OpenMu] = L1MuEta[i];
      L1OpenMuPhi[NL1OpenMu] = L1MuPhi[i];
      L1OpenMuIsol[NL1OpenMu] = L1MuIsol[i];
      L1OpenMuMip[NL1OpenMu] = L1MuMip[i];
      L1OpenMuFor[NL1OpenMu] = L1MuFor[i];
      L1OpenMuRPC[NL1OpenMu] = L1MuRPC[i];
      L1OpenMuQal[NL1OpenMu] = L1MuQal[i];
      NL1OpenMu++;
    }
  }
  // init
  for (int i=0;i<10;i++) {
    NL1GoodSingleMu = 0;
    L1GoodSingleMuPt[i] = -999.;
    L1GoodSingleMuE[i] = -999.;
    L1GoodSingleMuEta[i] = -999.;
    L1GoodSingleMuPhi[i] = -999.;
    L1GoodSingleMuIsol[i] = -999;
    L1GoodSingleMuMip[i] = -999;
    L1GoodSingleMuFor[i] = -999;
    L1GoodSingleMuRPC[i] = -999;
    L1GoodSingleMuQal[i] = -999;     
  }
  // Cut on muon quality      
  for (int i=0;i<NL1Mu;i++) {
    if ( L1MuQal[i]==4 || L1MuQal[i]==5 || L1MuQal[i]==6 || L1MuQal[i]==7 ) {
      L1GoodSingleMuPt[NL1GoodSingleMu] = L1MuPt[i];
      L1GoodSingleMuE[NL1GoodSingleMu] = L1MuE[i];
      L1GoodSingleMuEta[NL1GoodSingleMu] = L1MuEta[i];
      L1GoodSingleMuPhi[NL1GoodSingleMu] = L1MuPhi[i];
      L1GoodSingleMuIsol[NL1GoodSingleMu] = L1MuIsol[i];
      L1GoodSingleMuMip[NL1GoodSingleMu] = L1MuMip[i];
      L1GoodSingleMuFor[NL1GoodSingleMu] = L1MuFor[i];
      L1GoodSingleMuRPC[NL1GoodSingleMu] = L1MuRPC[i];
      L1GoodSingleMuQal[NL1GoodSingleMu] = L1MuQal[i];
      NL1GoodSingleMu++;
    }
  }

  // init
  for (int i=0;i<10;i++) {
    NL1GoodDoubleMu = 0;
    L1GoodDoubleMuPt[i] = -999.;
    L1GoodDoubleMuE[i] = -999.;
    L1GoodDoubleMuEta[i] = -999.;
    L1GoodDoubleMuPhi[i] = -999.;
    L1GoodDoubleMuIsol[i] = -999;
    L1GoodDoubleMuMip[i] = -999;
    L1GoodDoubleMuFor[i] = -999;
    L1GoodDoubleMuRPC[i] = -999;
    L1GoodDoubleMuQal[i] = -999;     
  }
  // Cut on muon quality
  for (int i=0;i<NL1Mu;i++) {
    if ( L1MuQal[i]==3 || L1MuQal[i]==5 || L1MuQal[i]==6 || L1MuQal[i]==7 ) {
      L1GoodDoubleMuPt[NL1GoodDoubleMu] = L1MuPt[i];
      L1GoodDoubleMuE[NL1GoodDoubleMu] = L1MuE[i];
      L1GoodDoubleMuEta[NL1GoodDoubleMu] = L1MuEta[i];
      L1GoodDoubleMuPhi[NL1GoodDoubleMu] = L1MuPhi[i];
      L1GoodDoubleMuIsol[NL1GoodDoubleMu] = L1MuIsol[i];
      L1GoodDoubleMuMip[NL1GoodDoubleMu] = L1MuMip[i];
      L1GoodDoubleMuFor[NL1GoodDoubleMu] = L1MuFor[i];
      L1GoodDoubleMuRPC[NL1GoodDoubleMu] = L1MuRPC[i];
      L1GoodDoubleMuQal[NL1GoodDoubleMu] = L1MuQal[i];
      NL1GoodDoubleMu++;
    }
  }
}

void OHltTree::SetOpenL1Bits()
{
  OpenL1_ZeroBias = 1;
  map_BitOfStandardHLTPath["OpenL1_ZeroBias"] = OpenL1_ZeroBias;

  if(L1GoodSingleMuPt[0] > 3.0 && (L1NIsolEmEt[0] > 5.0 || L1IsolEmEt[0] > 5.0)) 
    OpenL1_Mu3EG5 = 1; 
  else 
    OpenL1_Mu3EG5 = 0; 

}


Bool_t OHltTree::Notify()
{
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}
#endif // #ifdef OHltTree_cxx

#endif // #ifdef OHltTree_h
