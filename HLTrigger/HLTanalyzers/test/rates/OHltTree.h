//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Wed Dec 12 10:37:25 2007 by ROOT version 5.14/00f
// from TTree HltTree/
// found on file: /tmp/test.root
//
// Adapted to hold methods for OpenHLT rate counting
// Author: Vladimir Rekovic (UMN)   Dec, 2007
//////////////////////////////////////////////////////////

#ifndef OHltTree_h
#define OHltTree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

#include <vector>
#include <map>

class OHltTree {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leave types
   Int_t           NrecoJetCal;
   Int_t           NrecoJetGen;
   Int_t           NrecoTowCal;
   Float_t         recoJetCalPt[50];   //[NrecoJetCal]
   Float_t         recoJetCalPhi[50];   //[NrecoJetCal]
   Float_t         recoJetCalEta[50];   //[NrecoJetCal]
   Float_t         recoJetCalEt[50];   //[NrecoJetCal]
   Float_t         recoJetCalE[50];   //[NrecoJetCal]
   Float_t         recoJetGenPt[1];   //[NrecoJetGen]
   Float_t         recoJetGenPhi[1];   //[NrecoJetGen]
   Float_t         recoJetGenEta[1];   //[NrecoJetGen]
   Float_t         recoJetGenEt[1];   //[NrecoJetGen]
   Float_t         recoJetGenE[1];   //[NrecoJetGen]
   Float_t         recoTowEt[1];   //[NrecoTowCal]
   Float_t         recoTowEta[1];   //[NrecoTowCal]
   Float_t         recoTowPhi[1];   //[NrecoTowCal]
   Float_t         recoTowE[1];   //[NrecoTowCal]
   Float_t         recoTowEm[1];   //[NrecoTowCal]
   Float_t         recoTowHad[1];   //[NrecoTowCal]
   Float_t         recoTowOE[1];   //[NrecoTowCal]
   Float_t         recoMetCal;
   Float_t         recoMetCalPhi;
   Float_t         recoMetCalSum;
   Float_t         recoMetGen;
   Float_t         recoMetGenPhi;
   Float_t         recoMetGenSum;
   Float_t         recoHTCal;
   Float_t         recoHTCalPhi;
   Float_t         recoHTCalSum;
   Int_t           NohTau1;
   Float_t         ohTau1Eta[4];   //[NohTau1]
   Float_t         ohTau1Phi[4];   //[NohTau1]
   Float_t         ohTau1Pt[4];   //[NohTau1]
   Float_t         ohTau1Eiso[4];   //[NohTau1]
   Float_t         ohTau1L25Tpt[4];   //[NohTau1]
   Int_t           ohTau1L25Tiso[4];   //[NohTau1]
   Float_t         ohTau1L3Tpt[4];   //[NohTau1]
   Int_t           ohTau1L3Tiso[4];   //[NohTau1]
   Int_t           NohBJetLife;
   Float_t         ohBJetLifeL2E[4];   //[NohBJetLife]
   Float_t         ohBJetLifeL2ET[4];   //[NohBJetLife]
   Float_t         ohBJetLifeL2Eta[4];   //[NohBJetLife]
   Float_t         ohBJetLifeL2Phi[4];   //[NohBJetLife]
   Float_t         ohBJetLifeL25Discriminator[4];   //[NohBJetLife]
   Float_t         ohBJetLifeL3Discriminator[4];   //[NohBJetLife]
   Int_t           NohBJetPixelTracks;
   Float_t         ohBJetLifePixelTrackPt[100];   //[NohBJetPixelTracks]
   Float_t         ohBJetLifePixelTrackEta[100];   //[NohBJetPixelTracks]
   Float_t         ohBJetLifePixelTrackPhi[100];   //[NohBJetPixelTracks]
   Float_t         ohBJetLifePixelTrackChi2[100];   //[NohBJetPixelTracks]
   Int_t           NohBJetRegionalTracks;
   Float_t         ohBJetLifeRegionalTrackPt[100];   //[NohBJetRegionalTracks]
   Float_t         ohBJetLifeRegionalTrackEta[100];   //[NohBJetRegionalTracks]
   Float_t         ohBJetLifeRegionalTrackPhi[100];   //[NohBJetRegionalTracks]
   Float_t         ohBJetLifeRegionalTrackChi2[100];   //[NohBJetRegionalTracks]
   Float_t         ohBJetLifeRegionalSeedPt[100];   //[NohBJetRegionalTracks]
   Float_t         ohBJetLifeRegionalSeedEta[100];   //[NohBJetRegionalTracks]
   Float_t         ohBJetLifeRegionalSeedPhi[100];   //[NohBJetRegionalTracks]
   Int_t           NohBJetSoftm;
   Float_t         ohBJetSoftmL2E[2];   //[NohBJetSoftm]
   Float_t         ohBJetSoftmL2ET[2];   //[NohBJetSoftm]
   Float_t         ohBJetSoftmL2Eta[2];   //[NohBJetSoftm]
   Float_t         ohBJetSoftmL2Phi[2];   //[NohBJetSoftm]
   Float_t         ohBJetSoftmL25Discriminator[2];   //[NohBJetSoftm]
   Float_t         ohBJetSoftmL3Discriminator[2];   //[NohBJetSoftm]
   Int_t           NohBJetPerf;
   Float_t         ohBJetPerfL2E[2];   //[NohBJetPerf]
   Float_t         ohBJetPerfL2ET[2];   //[NohBJetPerf]
   Float_t         ohBJetPerfL2Eta[2];   //[NohBJetPerf]
   Float_t         ohBJetPerfL2Phi[2];   //[NohBJetPerf]
   Float_t         ohBJetPerfL25Discriminator[2];   //[NohBJetPerf]
   Float_t         ohBJetPerfL3Discriminator[2];   //[NohBJetPerf]
   Int_t           NrecoElec;
   Float_t         recoElecPt[1];   //[NrecoElec]
   Float_t         recoElecPhi[1];   //[NrecoElec]
   Float_t         recoElecEta[1];   //[NrecoElec]
   Float_t         recoElecEt[1];   //[NrecoElec]
   Float_t         recoElecE[1];   //[NrecoElec]
   Int_t           NrecoPhot;
   Float_t         recoPhotPt[1];   //[NrecoPhot]
   Float_t         recoPhotPhi[1];   //[NrecoPhot]
   Float_t         recoPhotEta[1];   //[NrecoPhot]
   Float_t         recoPhotEt[1];   //[NrecoPhot]
   Float_t         recoPhotE[1];   //[NrecoPhot]
   Int_t           NohPhot;
   Float_t         ohPhotEt[15];   //[NohPhot]
   Float_t         ohPhotEta[15];   //[NohPhot]
   Float_t         ohPhotPhi[15];   //[NohPhot]
   Float_t         ohPhotEiso[15];   //[NohPhot]
   Float_t         ohPhotHiso[15];   //[NohPhot]
   Float_t         ohPhotTiso[15];   //[NohPhot]
   Int_t           ohPhotL1iso[15];   //[NohPhot]
   Int_t           NohEle;
   Float_t         ohEleEt[21];   //[NohEle]
   Float_t         ohEleEta[21];   //[NohEle]
   Float_t         ohElePhi[21];   //[NohEle]
   Float_t         ohEleE[21];   //[NohEle]
   Float_t         ohEleP[21];   //[NohEle]
   Float_t         ohEleHiso[21];   //[NohEle]
   Float_t         ohEleTiso[21];   //[NohEle]
   Int_t           ohEleL1iso[21];   //[NohEle]
   Int_t           ohElePixelSeeds[21];   //[NohEle]
   Int_t           ohEleNewSC[21];   //[NohEle]
   Int_t           NohEleLW;
   Float_t         ohEleEtLW[53];   //[NohEleLW]
   Float_t         ohEleEtaLW[53];   //[NohEleLW]
   Float_t         ohElePhiLW[53];   //[NohEleLW]
   Float_t         ohEleELW[53];   //[NohEleLW]
   Float_t         ohElePLW[53];   //[NohEleLW]
   Float_t         ohEleHisoLW[53];   //[NohEleLW]
   Float_t         ohEleTisoLW[53];   //[NohEleLW]
   Int_t           ohEleL1isoLW[53];   //[NohEleLW]
   Int_t           ohElePixelSeedsLW[53];   //[NohEleLW]
   Int_t           ohEleNewSCLW[53];   //[NohEleLW]
   Int_t           NrecoMuon;
   Float_t         recoMuonPt[1];   //[NrecoMuon]
   Float_t         recoMuonPhi[1];   //[NrecoMuon]
   Float_t         recoMuonEta[1];   //[NrecoMuon]
   Float_t         recoMuonEt[1];   //[NrecoMuon]
   Float_t         recoMuonE[1];   //[NrecoMuon]
   Int_t           NohMuL2;
   Float_t         ohMuL2Pt[4];   //[NohMuL2]
   Float_t         ohMuL2Phi[4];   //[NohMuL2]
   Float_t         ohMuL2Eta[4];   //[NohMuL2]
   Int_t           ohMuL2Chg[4];   //[NohMuL2]
   Float_t         ohMuL2PtErr[4];   //[NohMuL2]
   Int_t           ohMuL2Iso[4];   //[NohMuL2]
   Float_t         ohMuL2Dr[4];   //[NohMuL2]
   Float_t         ohMuL2Dz[4];   //[NohMuL2]
   Int_t           NohMuL3;
   Float_t         ohMuL3Pt[4];   //[NohMuL3]
   Float_t         ohMuL3Phi[4];   //[NohMuL3]
   Float_t         ohMuL3Eta[4];   //[NohMuL3]
   Int_t           ohMuL3Chg[4];   //[NohMuL3]
   Float_t         ohMuL3PtErr[4];   //[NohMuL3]
   Int_t           ohMuL3Iso[4];   //[NohMuL3]
   Float_t         ohMuL3Dr[4];   //[NohMuL3]
   Float_t         ohMuL3Dz[4];   //[NohMuL3]
   Int_t           ohMuL3L2idx[4];   //[NohMuL3]
   Int_t           NMCpart;
   Int_t           MCpid[1539];   //[NMCpart]
   Int_t           MCstatus[1539];   //[NMCpart]
   Float_t         MCvtxX[1539];   //[NMCpart]
   Float_t         MCvtxY[1539];   //[NMCpart]
   Float_t         MCvtxZ[1539];   //[NMCpart]
   Float_t         MCpt[1539];   //[NMCpart]
   Float_t         MCeta[1539];   //[NMCpart]
   Float_t         MCphi[1539];   //[NMCpart]
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
   Float_t         L1IsolEmEt[7];   //[NL1IsolEm]
   Float_t         L1IsolEmE[7];   //[NL1IsolEm]
   Float_t         L1IsolEmEta[7];   //[NL1IsolEm]
   Float_t         L1IsolEmPhi[7];   //[NL1IsolEm]
   Int_t           NL1NIsolEm;
   Float_t         L1NIsolEmEt[7];   //[NL1NIsolEm]
   Float_t         L1NIsolEmE[7];   //[NL1NIsolEm]
   Float_t         L1NIsolEmEta[7];   //[NL1NIsolEm]
   Float_t         L1NIsolEmPhi[7];   //[NL1NIsolEm]
   Int_t           NL1Mu;
   Float_t         L1MuPt[7];   //[NL1Mu]
   Float_t         L1MuE[7];   //[NL1Mu]
   Float_t         L1MuEta[7];   //[NL1Mu]
   Float_t         L1MuPhi[7];   //[NL1Mu]
   Int_t           L1MuIsol[7];   //[NL1Mu]
   Int_t           L1MuMip[7];   //[NL1Mu]
   Int_t           L1MuFor[7];   //[NL1Mu]
   Int_t           L1MuRPC[7];   //[NL1Mu]
   Int_t           L1MuQal[7];   //[NL1Mu]
   Int_t           NL1CenJet;
   Float_t         L1CenJetEt[7];   //[NL1CenJet]
   Float_t         L1CenJetE[7];   //[NL1CenJet]
   Float_t         L1CenJetEta[7];   //[NL1CenJet]
   Float_t         L1CenJetPhi[7];   //[NL1CenJet]
   Int_t           NL1ForJet;
   Float_t         L1ForJetEt[7];   //[NL1ForJet]
   Float_t         L1ForJetE[7];   //[NL1ForJet]
   Float_t         L1ForJetEta[7];   //[NL1ForJet]
   Float_t         L1ForJetPhi[7];   //[NL1ForJet]
   Int_t           NL1Tau;
   Float_t         L1TauEt[7];   //[NL1Tau]
   Float_t         L1TauE[7];   //[NL1Tau]
   Float_t         L1TauEta[7];   //[NL1Tau]
   Float_t         L1TauPhi[7];   //[NL1Tau]
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
   Int_t           HLT1jet;
   Int_t           HLT2jet;
   Int_t           HLT3jet;
   Int_t           HLT4jet;
   Int_t           HLT1MET;
   Int_t           HLT2jetAco;
   Int_t           HLT1jet1METAco;
   Int_t           HLT1jet1MET;
   Int_t           HLT2jet1MET;
   Int_t           HLT3jet1MET;
   Int_t           HLT4jet1MET;
   Int_t           HLT1MET1HT;
   Int_t           CandHLT1SumET;
   Int_t           HLT1jetPE1;
   Int_t           HLT1jetPE3;
   Int_t           HLT1jetPE5;
   Int_t           HLT1jetPE7;
   Int_t           HLT1METPre1;
   Int_t           HLT1METPre2;
   Int_t           HLT1METPre3;
   Int_t           HLT2jetAve30;
   Int_t           HLT2jetAve60;
   Int_t           HLT2jetAve110;
   Int_t           HLT2jetAve150;
   Int_t           HLT2jetAve200;
   Int_t           HLT2jetvbfMET;
   Int_t           HLTS2jet1METNV;
   Int_t           HLTS2jet1METAco;
   Int_t           HLTSjet1MET1Aco;
   Int_t           HLTSjet2MET1Aco;
   Int_t           HLTS2jetMET1Aco;
   Int_t           HLTJetMETRapidityGap;
   Int_t           HLT1Electron;
   Int_t           HLT1ElectronRelaxed;
   Int_t           HLT2Electron;
   Int_t           HLT2ElectronRelaxed;
   Int_t           HLT1Photon;
   Int_t           HLT1PhotonRelaxed;
   Int_t           HLT2Photon;
   Int_t           HLT2PhotonRelaxed;
   Int_t           HLT1EMHighEt;
   Int_t           HLT1EMVeryHighEt;
   Int_t           HLT2ElectronZCounter;
   Int_t           HLT2ElectronExclusive;
   Int_t           HLT2PhotonExclusive;
   Int_t           HLT1PhotonL1Isolated;
   Int_t           CandHLT1ElectronStartup;
   Int_t           CandHLT1ElectronRelaxedStartup;
   Int_t           CandHLT2ElectronStartup;
   Int_t           CandHLT2ElectronRelaxedStartup;
   Int_t           HLT1MuonIso;
   Int_t           HLT1MuonNonIso;
   Int_t           HLT2MuonIso;
   Int_t           HLT2MuonNonIso;
   Int_t           HLT2MuonJPsi;
   Int_t           HLT2MuonUpsilon;
   Int_t           HLT2MuonZ;
   Int_t           HLTNMuonNonIso;
   Int_t           HLT2MuonSameSign;
   Int_t           HLT1MuonPrescalePt3;
   Int_t           HLT1MuonPrescalePt5;
   Int_t           HLT1MuonPrescalePt7x7;
   Int_t           HLT1MuonPrescalePt7x10;
   Int_t           HLT1MuonLevel1;
   Int_t           CandHLT1MuonPrescaleVtx2cm;
   Int_t           CandHLT1MuonPrescaleVtx2mm;
   Int_t           CandHLT2MuonPrescaleVtx2cm;
   Int_t           CandHLT2MuonPrescaleVtx2mm;
   Int_t           HLTB1Jet;
   Int_t           HLTB2Jet;
   Int_t           HLTB3Jet;
   Int_t           HLTB4Jet;
   Int_t           HLTBHT;
   Int_t           HLTB1JetMu;
   Int_t           HLTB2JetMu;
   Int_t           HLTB3JetMu;
   Int_t           HLTB4JetMu;
   Int_t           HLTBHTMu;
   Int_t           HLTBJPsiMuMu;
   Int_t           HLT1Tau;
   Int_t           HLT1Tau1MET;
   Int_t           HLT2TauPixel;
   Int_t           HLTXElectronBJet;
   Int_t           HLTXMuonBJet;
   Int_t           HLTXMuonBJetSoftMuon;
   Int_t           HLTXElectron1Jet;
   Int_t           HLTXElectron2Jet;
   Int_t           HLTXElectron3Jet;
   Int_t           HLTXElectron4Jet;
   Int_t           HLTXMuonJets;
   Int_t           CandHLTXMuonNoL2IsoJets;
   Int_t           CandHLTXMuonNoIsoJets;
   Int_t           HLTXElectronMuon;
   Int_t           HLTXElectronMuonRelaxed;
   Int_t           HLTXElectronTau;
   Int_t           CandHLTXElectronTauPixel;
   Int_t           HLTXMuonTau;
   Int_t           CandHLTEcalPi0;
   Int_t           CandHLTEcalPhiSym;
   Int_t           CandHLTHcalPhiSym;
   Int_t           HLTHcalIsolatedTrack;
   Int_t           CandHLTHcalIsolatedTrackNoEcalIsol;
   Int_t           HLTMinBiasPixel;
   Int_t           CandHLTMinBiasForAlignment;
   Int_t           HLTMinBias;
   Int_t           HLTZeroBias;
   Int_t           HLTriggerType;
   Int_t           TriggerFinalPath;
   Int_t           L1_DoubleEG10;
   Int_t           L1_DoubleEG10_ETM20;
   Int_t           L1_DoubleEG10_HTT200;
   Int_t           L1_DoubleEG10_Mu3;
   Int_t           L1_DoubleEG15;
   Int_t           L1_DoubleEG5;
   Int_t           L1_DoubleIsoEG10;
   Int_t           L1_DoubleIsoEG5_ETM20;
   Int_t           L1_DoubleIsoEG5_HTT200;
   Int_t           L1_DoubleIsoEG5_Mu3;
   Int_t           L1_DoubleIsoEG8;
   Int_t           L1_DoubleJet100;
   Int_t           L1_DoubleJet50_ETM20;
   Int_t           L1_DoubleJet50_HTT200;
   Int_t           L1_DoubleJet70;
   Int_t           L1_DoubleMu3;
   Int_t           L1_DoubleMu3_EG10;
   Int_t           L1_DoubleMu3_ETM20;
   Int_t           L1_DoubleMu3_HTT200;
   Int_t           L1_DoubleMu3_IsoEG5;
   Int_t           L1_DoubleTauJet20;
   Int_t           L1_DoubleTauJet30;
   Int_t           L1_DoubleTauJet35;
   Int_t           L1_DoubleTauJet40;
   Int_t           L1_DoubleTauJet40_ETM20;
   Int_t           L1_DoubleTauJet40_HTT200;
   Int_t           L1_EG10_Jet15;
   Int_t           L1_EG12_ETM30;
   Int_t           L1_EG12_HTT200;
   Int_t           L1_EG12_Jet70;
   Int_t           L1_EG12_TauJet40;
   Int_t           L1_ETM10;
   Int_t           L1_ETM15;
   Int_t           L1_ETM20;
   Int_t           L1_ETM40;
   Int_t           L1_ETM45;
   Int_t           L1_ETM50;
   Int_t           L1_ETM60;
   Int_t           L1_ETT60;
   Int_t           L1_ExclusiveDoubleIsoEG6;
   Int_t           L1_ExclusiveDoubleJet60;
   Int_t           L1_ExclusiveJet25_Gap_Jet25;
   Int_t           L1_HTT100;
   Int_t           L1_HTT100_ETM30;
   Int_t           L1_HTT200;
   Int_t           L1_HTT250;
   Int_t           L1_HTT300;
   Int_t           L1_HTT400;
   Int_t           L1_HTT500;
   Int_t           L1_IsoEG10_EG10;
   Int_t           L1_IsoEG10_HTT200;
   Int_t           L1_IsoEG10_Jet15;
   Int_t           L1_IsoEG10_Jet15_ForJet10;
   Int_t           L1_IsoEG10_Jet20;
   Int_t           L1_IsoEG10_Jet30;
   Int_t           L1_IsoEG10_Jet70;
   Int_t           L1_IsoEG10_TauJet20;
   Int_t           L1_IsoEG10_TauJet30;
   Int_t           L1_Jet70_ETM40;
   Int_t           L1_Jet70_HTT200;
   Int_t           L1_Jet70_TauJet40;
   Int_t           L1_MinBias_HTT10;
   Int_t           L1_Mu3_EG12;
   Int_t           L1_Mu3_ETM30;
   Int_t           L1_Mu3_HTT200;
   Int_t           L1_Mu3_IsoEG5;
   Int_t           L1_Mu3_Jet15;
   Int_t           L1_Mu3_Jet70;
   Int_t           L1_Mu5_IsoEG10;
   Int_t           L1_Mu5_Jet15;
   Int_t           L1_Mu5_Jet20;
   Int_t           L1_Mu5_TauJet20;
   Int_t           L1_Mu5_TauJet30;
   Int_t           L1_QuadJet40;
   Int_t           L1_SingleEG10;
   Int_t           L1_SingleEG12;
   Int_t           L1_SingleEG15;
   Int_t           L1_SingleEG20;
   Int_t           L1_SingleEG25;
   Int_t           L1_SingleEG5;
   Int_t           L1_SingleEG8;
   Int_t           L1_SingleIsoEG10;
   Int_t           L1_SingleIsoEG12;
   Int_t           L1_SingleIsoEG15;
   Int_t           L1_SingleIsoEG20;
   Int_t           L1_SingleIsoEG25;
   Int_t           L1_SingleIsoEG5;
   Int_t           L1_SingleIsoEG8;
   Int_t           L1_SingleJet100;
   Int_t           L1_SingleJet15;
   Int_t           L1_SingleJet150;
   Int_t           L1_SingleJet20;
   Int_t           L1_SingleJet200;
   Int_t           L1_SingleJet30;
   Int_t           L1_SingleJet50;
   Int_t           L1_SingleJet70;
   Int_t           L1_SingleMu10;
   Int_t           L1_SingleMu14;
   Int_t           L1_SingleMu20;
   Int_t           L1_SingleMu25;
   Int_t           L1_SingleMu3;
   Int_t           L1_SingleMu5;
   Int_t           L1_SingleMu7;
   Int_t           L1_SingleTauJet10;
   Int_t           L1_SingleTauJet100;
   Int_t           L1_SingleTauJet20;
   Int_t           L1_SingleTauJet30;
   Int_t           L1_SingleTauJet40;
   Int_t           L1_SingleTauJet60;
   Int_t           L1_SingleTauJet80;
   Int_t           L1_TauJet20_ETM20;
   Int_t           L1_TauJet30_ETM30;
   Int_t           L1_TauJet30_ETM40;
   Int_t           L1_TauJet40_HTT200;
   Int_t           L1_TripleEG10;
   Int_t           L1_TripleIsoEG5;
   Int_t           L1_TripleJet50;
   Int_t           L1_TripleMu3;
   Int_t           L1_TripleTauJet40;
   Int_t           L1_VBF_DoubleTauHad;
   Int_t           L1_VBF_ETM50;
   Int_t           L1_VBF_ETM50_veto;
   Int_t           L1_VBF_IsoEG10_Tau_TauHad;
   Int_t           L1_VBF_IsoEG15;
   Int_t           L1_VBF_Mu10;
   Int_t           L1_VBF_Mu7_Tau_TauHad;
   Int_t           L1_VBF_QuadJet;
   Int_t           L1_ZeroBias;
   Int_t	L1_SingleMuOpen;
   Int_t	L1_SingleMuBeamHalo;
   Int_t	L1_SingleEG2;
   Int_t	L1_DoubleEG1;
   Int_t	L1_QuadJet15;
   Int_t	L1_QuadJet30;
   Int_t	L1_ETM30;
   Int_t	L1_EG5_TripleJet15;
   Int_t	L1_Mu3_TripleJet15;
   Int_t	L1_SingleJetCountsHFTow;
   Int_t	L1_DoubleJetCountsHFTow;
   Int_t	L1_SingleJetCountsHFRing0Sum3;
   Int_t	L1_DoubleJetCountsHFRing0Sum3;
   Int_t	L1_SingleJetCountsHFRing0Sum6;

   // List of branches
   TBranch        *b_NrecoJetCal;   //!
   TBranch        *b_NrecoJetGen;   //!
   TBranch        *b_NrecoTowCal;   //!
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
   TBranch        *b_NohTau1;   //!
   TBranch        *b_ohTau1Eta;   //!
   TBranch        *b_ohTau1Phi;   //!
   TBranch        *b_ohTau1Pt;   //!
   TBranch        *b_ohTau1Eiso;   //!
   TBranch        *b_ohTau1L25Tpt;   //!
   TBranch        *b_ohTau1L25Tiso;   //!
   TBranch        *b_ohTau1L3Tpt;   //!
   TBranch        *b_ohTau1L3Tiso;   //!
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
   TBranch        *b_NohBJetPerf;   //!
   TBranch        *b_ohBJetPerfL2E;   //!
   TBranch        *b_ohBJetPerfL2ET;   //!
   TBranch        *b_ohBJetPerfL2Eta;   //!
   TBranch        *b_ohBJetPerfL2Phi;   //!
   TBranch        *b_ohBJetPerfL25Discriminator;   //!
   TBranch        *b_ohBJetPerfL3Discriminator;   //!
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
   TBranch        *b_L1_DoubleEG10;   //!
   TBranch        *b_L1_DoubleEG10_ETM20;   //!
   TBranch        *b_L1_DoubleEG10_HTT200;   //!
   TBranch        *b_L1_DoubleEG10_Mu3;   //!
   TBranch        *b_L1_DoubleEG15;   //!
   TBranch        *b_L1_DoubleEG5;   //!
   TBranch        *b_L1_DoubleIsoEG10;   //!
   TBranch        *b_L1_DoubleIsoEG5_ETM20;   //!
   TBranch        *b_L1_DoubleIsoEG5_HTT200;   //!
   TBranch        *b_L1_DoubleIsoEG5_Mu3;   //!
   TBranch        *b_L1_DoubleIsoEG8;   //!
   TBranch        *b_L1_DoubleJet100;   //!
   TBranch        *b_L1_DoubleJet50_ETM20;   //!
   TBranch        *b_L1_DoubleJet50_HTT200;   //!
   TBranch        *b_L1_DoubleJet70;   //!
   TBranch        *b_L1_DoubleMu3;   //!
   TBranch        *b_L1_DoubleMu3_EG10;   //!
   TBranch        *b_L1_DoubleMu3_ETM20;   //!
   TBranch        *b_L1_DoubleMu3_HTT200;   //!
   TBranch        *b_L1_DoubleMu3_IsoEG5;   //!
   TBranch        *b_L1_DoubleTauJet20;   //!
   TBranch        *b_L1_DoubleTauJet30;   //!
   TBranch        *b_L1_DoubleTauJet35;   //!
   TBranch        *b_L1_DoubleTauJet40;   //!
   TBranch        *b_L1_DoubleTauJet40_ETM20;   //!
   TBranch        *b_L1_DoubleTauJet40_HTT200;   //!
   TBranch        *b_L1_EG10_Jet15;   //!
   TBranch        *b_L1_EG12_ETM30;   //!
   TBranch        *b_L1_EG12_HTT200;   //!
   TBranch        *b_L1_EG12_Jet70;   //!
   TBranch        *b_L1_EG12_TauJet40;   //!
   TBranch        *b_L1_ETM10;   //!
   TBranch        *b_L1_ETM15;   //!
   TBranch        *b_L1_ETM20;   //!
   TBranch        *b_L1_ETM40;   //!
   TBranch        *b_L1_ETM45;   //!
   TBranch        *b_L1_ETM50;   //!
   TBranch        *b_L1_ETM60;   //!
   TBranch        *b_L1_ETT60;   //!
   TBranch        *b_L1_ExclusiveDoubleIsoEG6;   //!
   TBranch        *b_L1_ExclusiveDoubleJet60;   //!
   TBranch        *b_L1_ExclusiveJet25_Gap_Jet25;   //!
   TBranch        *b_L1_HTT100;   //!
   TBranch        *b_L1_HTT100_ETM30;   //!
   TBranch        *b_L1_HTT200;   //!
   TBranch        *b_L1_HTT250;   //!
   TBranch        *b_L1_HTT300;   //!
   TBranch        *b_L1_HTT400;   //!
   TBranch        *b_L1_HTT500;   //!
   TBranch        *b_L1_IsoEG10_EG10;   //!
   TBranch        *b_L1_IsoEG10_HTT200;   //!
   TBranch        *b_L1_IsoEG10_Jet15;   //!
   TBranch        *b_L1_IsoEG10_Jet15_ForJet10;   //!
   TBranch        *b_L1_IsoEG10_Jet20;   //!
   TBranch        *b_L1_IsoEG10_Jet30;   //!
   TBranch        *b_L1_IsoEG10_Jet70;   //!
   TBranch        *b_L1_IsoEG10_TauJet20;   //!
   TBranch        *b_L1_IsoEG10_TauJet30;   //!
   TBranch        *b_L1_Jet70_ETM40;   //!
   TBranch        *b_L1_Jet70_HTT200;   //!
   TBranch        *b_L1_Jet70_TauJet40;   //!
   TBranch        *b_L1_MinBias_HTT10;   //!
   TBranch        *b_L1_Mu3_EG12;   //!
   TBranch        *b_L1_Mu3_ETM30;   //!
   TBranch        *b_L1_Mu3_HTT200;   //!
   TBranch        *b_L1_Mu3_IsoEG5;   //!
   TBranch        *b_L1_Mu3_Jet15;   //!
   TBranch        *b_L1_Mu3_Jet70;   //!
   TBranch        *b_L1_Mu5_IsoEG10;   //!
   TBranch        *b_L1_Mu5_Jet15;   //!
   TBranch        *b_L1_Mu5_Jet20;   //!
   TBranch        *b_L1_Mu5_TauJet20;   //!
   TBranch        *b_L1_Mu5_TauJet30;   //!
   TBranch        *b_L1_QuadJet40;   //!
   TBranch        *b_L1_SingleEG10;   //!
   TBranch        *b_L1_SingleEG12;   //!
   TBranch        *b_L1_SingleEG15;   //!
   TBranch        *b_L1_SingleEG20;   //!
   TBranch        *b_L1_SingleEG25;   //!
   TBranch        *b_L1_SingleEG5;   //!
   TBranch        *b_L1_SingleEG8;   //!
   TBranch        *b_L1_SingleIsoEG10;   //!
   TBranch        *b_L1_SingleIsoEG12;   //!
   TBranch        *b_L1_SingleIsoEG15;   //!
   TBranch        *b_L1_SingleIsoEG20;   //!
   TBranch        *b_L1_SingleIsoEG25;   //!
   TBranch        *b_L1_SingleIsoEG5;   //!
   TBranch        *b_L1_SingleIsoEG8;   //!
   TBranch        *b_L1_SingleJet100;   //!
   TBranch        *b_L1_SingleJet15;   //!
   TBranch        *b_L1_SingleJet150;   //!
   TBranch        *b_L1_SingleJet20;   //!
   TBranch        *b_L1_SingleJet200;   //!
   TBranch        *b_L1_SingleJet30;   //!
   TBranch        *b_L1_SingleJet50;   //!
   TBranch        *b_L1_SingleJet70;   //!
   TBranch        *b_L1_SingleMu10;   //!
   TBranch        *b_L1_SingleMu14;   //!
   TBranch        *b_L1_SingleMu20;   //!
   TBranch        *b_L1_SingleMu25;   //!
   TBranch        *b_L1_SingleMu3;   //!
   TBranch        *b_L1_SingleMu5;   //!
   TBranch        *b_L1_SingleMu7;   //!
   TBranch        *b_L1_SingleTauJet10;   //!
   TBranch        *b_L1_SingleTauJet100;   //!
   TBranch        *b_L1_SingleTauJet20;   //!
   TBranch        *b_L1_SingleTauJet30;   //!
   TBranch        *b_L1_SingleTauJet40;   //!
   TBranch        *b_L1_SingleTauJet60;   //!
   TBranch        *b_L1_SingleTauJet80;   //!
   TBranch        *b_L1_TauJet20_ETM20;   //!
   TBranch        *b_L1_TauJet30_ETM30;   //!
   TBranch        *b_L1_TauJet30_ETM40;   //!
   TBranch        *b_L1_TauJet40_HTT200;   //!
   TBranch        *b_L1_TripleEG10;   //!
   TBranch        *b_L1_TripleIsoEG5;   //!
   TBranch        *b_L1_TripleJet50;   //!
   TBranch        *b_L1_TripleMu3;   //!
   TBranch        *b_L1_TripleTauJet40;   //!
   TBranch        *b_L1_VBF_DoubleTauHad;   //!
   TBranch        *b_L1_VBF_ETM50;   //!
   TBranch        *b_L1_VBF_ETM50_veto;   //!
   TBranch        *b_L1_VBF_IsoEG10_Tau_TauHad;   //!
   TBranch        *b_L1_VBF_IsoEG15;   //!
   TBranch        *b_L1_VBF_Mu10;   //!
   TBranch        *b_L1_VBF_Mu7_Tau_TauHad;   //!
   TBranch        *b_L1_VBF_QuadJet;   //!
   TBranch        *b_L1_ZeroBias;   //!
   TBranch        *b_L1_SingleMuOpen; 
   TBranch        *b_L1_SingleMuBeamHalo; 
   TBranch        *b_L1_SingleEG2; 
   TBranch        *b_L1_DoubleEG1; 
   TBranch        *b_L1_QuadJet15; 
   TBranch        *b_L1_QuadJet30; 
   TBranch        *b_L1_ETM30; 
   TBranch        *b_L1_EG5_TripleJet15; 
   TBranch        *b_L1_Mu3_TripleJet15; 
   TBranch        *b_L1_SingleJetCountsHFTow; 
   TBranch        *b_L1_DoubleJetCountsHFTow; 
   TBranch        *b_L1_SingleJetCountsHFRing0Sum3; 
   TBranch        *b_L1_DoubleJetCountsHFRing0Sum3; 
   TBranch        *b_L1_SingleJetCountsHFRing0Sum6; 

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

   OHltTree(TTree *tree=0,int ntrig=0);
   virtual ~OHltTree();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual Bool_t   Notify();
   virtual void	    SetMapBitOfStandardHLTPath();
   virtual void     Show(Long64_t entry = -1);

   void Loop(std::vector<int> *, std::vector<int> *, std::vector<int> * 
	     ,std::vector< std::vector<int> > * overlapCount
	     ,std::vector<TString> trignames, std::map<TString,int> map_TrigPrescls
	     ,int n=-1,bool doMuonCut=false,bool doElecCut=false, double muonPt=-999., double muonDr=-999.);
   void PrintOhltVariables(int level, int type);
   int OpenHltTauPassed(float Et,float Eiso, float L25Tpt, int L25Tiso,float L3Tpt, int L3Tiso);
   int OpenHlt1ElectronPassed(float Et,int L1iso,float Tiso,float Hiso);
   int OpenHlt1PhotonPassed(float Et,int L1iso,float Tiso,float Eiso,float HisoBR,float HisoEC);
   int OpenHlt1MuonPassed(double ptl1,double ptl2,double ptl3,double dr,int iso);
   int OpenHlt2MuonPassed(double ptl1,double ptl2,double ptl3,double dr,int iso);
   int OpenHlt1JetPassed(double pt);

 private:
   int Ntrig;
   std::vector<int> triggerBit;
   std::vector<int> triggerBitNoPrescale;
   std::vector<int> previousBitsFired;
   std::vector<int> allOtherBitsFired;
   std::vector<int> BitOfStandardHLTPath;
   std::map<TString,int> map_BitOfStandardHLTPath;
   
   enum e_objType {
     muon,
     electron,
     tau,
     photon,
     jet
   };
   
};

#endif

#ifdef OHltTree_cxx
OHltTree::OHltTree(TTree *tree, int ntrig)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/tmp/test.root");
      if (!f) {
         f = new TFile("/tmp/test.root");
      }
      tree = (TTree*)gDirectory->Get("HltTree");

   }
   Init(tree);

   Ntrig = ntrig;
   triggerBit.reserve(Ntrig);
   triggerBitNoPrescale.reserve(Ntrig);
   previousBitsFired.reserve(Ntrig);
   allOtherBitsFired.reserve(Ntrig);
   BitOfStandardHLTPath.reserve(Ntrig);

   for (int it = 0; it < Ntrig; it++){
     triggerBit.push_back(false);
     triggerBitNoPrescale.push_back(false);
     previousBitsFired.push_back(false);
     allOtherBitsFired.push_back(false);
   }
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
   // It is normaly not necessary to make changes to the generated
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
   fChain->SetBranchAddress("recoMetCal", &recoMetCal, &b_recoMetCal);
   fChain->SetBranchAddress("recoMetCalPhi", &recoMetCalPhi, &b_recoMetCalPhi);
   fChain->SetBranchAddress("recoMetCalSum", &recoMetCalSum, &b_recoMetCalSum);
   fChain->SetBranchAddress("recoMetGen", &recoMetGen, &b_recoMetGen);
   fChain->SetBranchAddress("recoMetGenPhi", &recoMetGenPhi, &b_recoMetGenPhi);
   fChain->SetBranchAddress("recoMetGenSum", &recoMetGenSum, &b_recoMetGenSum);
   fChain->SetBranchAddress("recoHTCal", &recoHTCal, &b_recoHTCal);
   fChain->SetBranchAddress("recoHTCalPhi", &recoHTCalPhi, &b_recoHTCalPhi);
   fChain->SetBranchAddress("recoHTCalSum", &recoHTCalSum, &b_recoHTCalSum);
   fChain->SetBranchAddress("NohTau1", &NohTau1, &b_NohTau1);
   fChain->SetBranchAddress("ohTau1Eta", ohTau1Eta, &b_ohTau1Eta);
   fChain->SetBranchAddress("ohTau1Phi", ohTau1Phi, &b_ohTau1Phi);
   fChain->SetBranchAddress("ohTau1Pt", ohTau1Pt, &b_ohTau1Pt);
   fChain->SetBranchAddress("ohTau1Eiso", ohTau1Eiso, &b_ohTau1Eiso);
   fChain->SetBranchAddress("ohTau1L25Tpt", ohTau1L25Tpt, &b_ohTau1L25Tpt);
   fChain->SetBranchAddress("ohTau1L25Tiso", ohTau1L25Tiso, &b_ohTau1L25Tiso);
   fChain->SetBranchAddress("ohTau1L3Tpt", ohTau1L3Tpt, &b_ohTau1L3Tpt);
   fChain->SetBranchAddress("ohTau1L3Tiso", ohTau1L3Tiso, &b_ohTau1L3Tiso);
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
   fChain->SetBranchAddress("NohBJetPerf", &NohBJetPerf, &b_NohBJetPerf);
   fChain->SetBranchAddress("ohBJetPerfL2E", ohBJetPerfL2E, &b_ohBJetPerfL2E);
   fChain->SetBranchAddress("ohBJetPerfL2ET", ohBJetPerfL2ET, &b_ohBJetPerfL2ET);
   fChain->SetBranchAddress("ohBJetPerfL2Eta", ohBJetPerfL2Eta, &b_ohBJetPerfL2Eta);
   fChain->SetBranchAddress("ohBJetPerfL2Phi", ohBJetPerfL2Phi, &b_ohBJetPerfL2Phi);
   fChain->SetBranchAddress("ohBJetPerfL25Discriminator", ohBJetPerfL25Discriminator, &b_ohBJetPerfL25Discriminator);
   fChain->SetBranchAddress("ohBJetPerfL3Discriminator", ohBJetPerfL3Discriminator, &b_ohBJetPerfL3Discriminator);
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
   fChain->SetBranchAddress("HLT1jet", &HLT1jet, &b_HLT1jet);
   fChain->SetBranchAddress("HLT2jet", &HLT2jet, &b_HLT2jet);
   fChain->SetBranchAddress("HLT3jet", &HLT3jet, &b_HLT3jet);
   fChain->SetBranchAddress("HLT4jet", &HLT4jet, &b_HLT4jet);
   fChain->SetBranchAddress("HLT1MET", &HLT1MET, &b_HLT1MET);
   fChain->SetBranchAddress("HLT2jetAco", &HLT2jetAco, &b_HLT2jetAco);
   fChain->SetBranchAddress("HLT1jet1METAco", &HLT1jet1METAco, &b_HLT1jet1METAco);
   fChain->SetBranchAddress("HLT1jet1MET", &HLT1jet1MET, &b_HLT1jet1MET);
   fChain->SetBranchAddress("HLT2jet1MET", &HLT2jet1MET, &b_HLT2jet1MET);
   fChain->SetBranchAddress("HLT3jet1MET", &HLT3jet1MET, &b_HLT3jet1MET);
   fChain->SetBranchAddress("HLT4jet1MET", &HLT4jet1MET, &b_HLT4jet1MET);
   fChain->SetBranchAddress("HLT1MET1HT", &HLT1MET1HT, &b_HLT1MET1HT);
   fChain->SetBranchAddress("CandHLT1SumET", &CandHLT1SumET, &b_CandHLT1SumET);
   fChain->SetBranchAddress("HLT1jetPE1", &HLT1jetPE1, &b_HLT1jetPE1);
   fChain->SetBranchAddress("HLT1jetPE3", &HLT1jetPE3, &b_HLT1jetPE3);
   fChain->SetBranchAddress("HLT1jetPE5", &HLT1jetPE5, &b_HLT1jetPE5);
   fChain->SetBranchAddress("HLT1jetPE7", &HLT1jetPE7, &b_HLT1jetPE7);
   fChain->SetBranchAddress("HLT1METPre1", &HLT1METPre1, &b_HLT1METPre1);
   fChain->SetBranchAddress("HLT1METPre2", &HLT1METPre2, &b_HLT1METPre2);
   fChain->SetBranchAddress("HLT1METPre3", &HLT1METPre3, &b_HLT1METPre3);
   fChain->SetBranchAddress("HLT2jetAve30", &HLT2jetAve30, &b_HLT2jetAve30);
   fChain->SetBranchAddress("HLT2jetAve60", &HLT2jetAve60, &b_HLT2jetAve60);
   fChain->SetBranchAddress("HLT2jetAve110", &HLT2jetAve110, &b_HLT2jetAve110);
   fChain->SetBranchAddress("HLT2jetAve150", &HLT2jetAve150, &b_HLT2jetAve150);
   fChain->SetBranchAddress("HLT2jetAve200", &HLT2jetAve200, &b_HLT2jetAve200);
   fChain->SetBranchAddress("HLT2jetvbfMET", &HLT2jetvbfMET, &b_HLT2jetvbfMET);
   fChain->SetBranchAddress("HLTS2jet1METNV", &HLTS2jet1METNV, &b_HLTS2jet1METNV);
   fChain->SetBranchAddress("HLTS2jet1METAco", &HLTS2jet1METAco, &b_HLTS2jet1METAco);
   fChain->SetBranchAddress("HLTSjet1MET1Aco", &HLTSjet1MET1Aco, &b_HLTSjet1MET1Aco);
   fChain->SetBranchAddress("HLTSjet2MET1Aco", &HLTSjet2MET1Aco, &b_HLTSjet2MET1Aco);
   fChain->SetBranchAddress("HLTS2jetMET1Aco", &HLTS2jetMET1Aco, &b_HLTS2jetMET1Aco);
   fChain->SetBranchAddress("HLTJetMETRapidityGap", &HLTJetMETRapidityGap, &b_HLTJetMETRapidityGap);
   fChain->SetBranchAddress("HLT1Electron", &HLT1Electron, &b_HLT1Electron);
   fChain->SetBranchAddress("HLT1ElectronRelaxed", &HLT1ElectronRelaxed, &b_HLT1ElectronRelaxed);
   fChain->SetBranchAddress("HLT2Electron", &HLT2Electron, &b_HLT2Electron);
   fChain->SetBranchAddress("HLT2ElectronRelaxed", &HLT2ElectronRelaxed, &b_HLT2ElectronRelaxed);
   fChain->SetBranchAddress("HLT1Photon", &HLT1Photon, &b_HLT1Photon);
   fChain->SetBranchAddress("HLT1PhotonRelaxed", &HLT1PhotonRelaxed, &b_HLT1PhotonRelaxed);
   fChain->SetBranchAddress("HLT2Photon", &HLT2Photon, &b_HLT2Photon);
   fChain->SetBranchAddress("HLT2PhotonRelaxed", &HLT2PhotonRelaxed, &b_HLT2PhotonRelaxed);
   fChain->SetBranchAddress("HLT1EMHighEt", &HLT1EMHighEt, &b_HLT1EMHighEt);
   fChain->SetBranchAddress("HLT1EMVeryHighEt", &HLT1EMVeryHighEt, &b_HLT1EMVeryHighEt);
   fChain->SetBranchAddress("HLT2ElectronZCounter", &HLT2ElectronZCounter, &b_HLT2ElectronZCounter);
   fChain->SetBranchAddress("HLT2ElectronExclusive", &HLT2ElectronExclusive, &b_HLT2ElectronExclusive);
   fChain->SetBranchAddress("HLT2PhotonExclusive", &HLT2PhotonExclusive, &b_HLT2PhotonExclusive);
   fChain->SetBranchAddress("HLT1PhotonL1Isolated", &HLT1PhotonL1Isolated, &b_HLT1PhotonL1Isolated);
   fChain->SetBranchAddress("CandHLT1ElectronStartup", &CandHLT1ElectronStartup, &b_CandHLT1ElectronStartup);
   fChain->SetBranchAddress("CandHLT1ElectronRelaxedStartup", &CandHLT1ElectronRelaxedStartup, &b_CandHLT1ElectronRelaxedStartup);
   fChain->SetBranchAddress("CandHLT2ElectronStartup", &CandHLT2ElectronStartup, &b_CandHLT2ElectronStartup);
   fChain->SetBranchAddress("CandHLT2ElectronRelaxedStartup", &CandHLT2ElectronRelaxedStartup, &b_CandHLT2ElectronRelaxedStartup);
   fChain->SetBranchAddress("HLT1MuonIso", &HLT1MuonIso, &b_HLT1MuonIso);
   fChain->SetBranchAddress("HLT1MuonNonIso", &HLT1MuonNonIso, &b_HLT1MuonNonIso);
   fChain->SetBranchAddress("HLT2MuonIso", &HLT2MuonIso, &b_HLT2MuonIso);
   fChain->SetBranchAddress("HLT2MuonNonIso", &HLT2MuonNonIso, &b_HLT2MuonNonIso);
   fChain->SetBranchAddress("HLT2MuonJPsi", &HLT2MuonJPsi, &b_HLT2MuonJPsi);
   fChain->SetBranchAddress("HLT2MuonUpsilon", &HLT2MuonUpsilon, &b_HLT2MuonUpsilon);
   fChain->SetBranchAddress("HLT2MuonZ", &HLT2MuonZ, &b_HLT2MuonZ);
   fChain->SetBranchAddress("HLTNMuonNonIso", &HLTNMuonNonIso, &b_HLTNMuonNonIso);
   fChain->SetBranchAddress("HLT2MuonSameSign", &HLT2MuonSameSign, &b_HLT2MuonSameSign);
   fChain->SetBranchAddress("HLT1MuonPrescalePt3", &HLT1MuonPrescalePt3, &b_HLT1MuonPrescalePt3);
   fChain->SetBranchAddress("HLT1MuonPrescalePt5", &HLT1MuonPrescalePt5, &b_HLT1MuonPrescalePt5);
   fChain->SetBranchAddress("HLT1MuonPrescalePt7x7", &HLT1MuonPrescalePt7x7, &b_HLT1MuonPrescalePt7x7);
   fChain->SetBranchAddress("HLT1MuonPrescalePt7x10", &HLT1MuonPrescalePt7x10, &b_HLT1MuonPrescalePt7x10);
   fChain->SetBranchAddress("HLT1MuonLevel1", &HLT1MuonLevel1, &b_HLT1MuonLevel1);
   fChain->SetBranchAddress("CandHLT1MuonPrescaleVtx2cm", &CandHLT1MuonPrescaleVtx2cm, &b_CandHLT1MuonPrescaleVtx2cm);
   fChain->SetBranchAddress("CandHLT1MuonPrescaleVtx2mm", &CandHLT1MuonPrescaleVtx2mm, &b_CandHLT1MuonPrescaleVtx2mm);
   fChain->SetBranchAddress("CandHLT2MuonPrescaleVtx2cm", &CandHLT2MuonPrescaleVtx2cm, &b_CandHLT2MuonPrescaleVtx2cm);
   fChain->SetBranchAddress("CandHLT2MuonPrescaleVtx2mm", &CandHLT2MuonPrescaleVtx2mm, &b_CandHLT2MuonPrescaleVtx2mm);
   fChain->SetBranchAddress("HLTB1Jet", &HLTB1Jet, &b_HLTB1Jet);
   fChain->SetBranchAddress("HLTB2Jet", &HLTB2Jet, &b_HLTB2Jet);
   fChain->SetBranchAddress("HLTB3Jet", &HLTB3Jet, &b_HLTB3Jet);
   fChain->SetBranchAddress("HLTB4Jet", &HLTB4Jet, &b_HLTB4Jet);
   fChain->SetBranchAddress("HLTBHT", &HLTBHT, &b_HLTBHT);
   fChain->SetBranchAddress("HLTB1JetMu", &HLTB1JetMu, &b_HLTB1JetMu);
   fChain->SetBranchAddress("HLTB2JetMu", &HLTB2JetMu, &b_HLTB2JetMu);
   fChain->SetBranchAddress("HLTB3JetMu", &HLTB3JetMu, &b_HLTB3JetMu);
   fChain->SetBranchAddress("HLTB4JetMu", &HLTB4JetMu, &b_HLTB4JetMu);
   fChain->SetBranchAddress("HLTBHTMu", &HLTBHTMu, &b_HLTBHTMu);
   fChain->SetBranchAddress("HLTBJPsiMuMu", &HLTBJPsiMuMu, &b_HLTBJPsiMuMu);
   fChain->SetBranchAddress("HLT1Tau", &HLT1Tau, &b_HLT1Tau);
   fChain->SetBranchAddress("HLT1Tau1MET", &HLT1Tau1MET, &b_HLT1Tau1MET);
   fChain->SetBranchAddress("HLT2TauPixel", &HLT2TauPixel, &b_HLT2TauPixel);
   fChain->SetBranchAddress("HLTXElectronBJet", &HLTXElectronBJet, &b_HLTXElectronBJet);
   fChain->SetBranchAddress("HLTXMuonBJet", &HLTXMuonBJet, &b_HLTXMuonBJet);
   fChain->SetBranchAddress("HLTXMuonBJetSoftMuon", &HLTXMuonBJetSoftMuon, &b_HLTXMuonBJetSoftMuon);
   fChain->SetBranchAddress("HLTXElectron1Jet", &HLTXElectron1Jet, &b_HLTXElectron1Jet);
   fChain->SetBranchAddress("HLTXElectron2Jet", &HLTXElectron2Jet, &b_HLTXElectron2Jet);
   fChain->SetBranchAddress("HLTXElectron3Jet", &HLTXElectron3Jet, &b_HLTXElectron3Jet);
   fChain->SetBranchAddress("HLTXElectron4Jet", &HLTXElectron4Jet, &b_HLTXElectron4Jet);
   fChain->SetBranchAddress("HLTXMuonJets", &HLTXMuonJets, &b_HLTXMuonJets);
   fChain->SetBranchAddress("CandHLTXMuonNoL2IsoJets", &CandHLTXMuonNoL2IsoJets, &b_CandHLTXMuonNoL2IsoJets);
   fChain->SetBranchAddress("CandHLTXMuonNoIsoJets", &CandHLTXMuonNoIsoJets, &b_CandHLTXMuonNoIsoJets);
   fChain->SetBranchAddress("HLTXElectronMuon", &HLTXElectronMuon, &b_HLTXElectronMuon);
   fChain->SetBranchAddress("HLTXElectronMuonRelaxed", &HLTXElectronMuonRelaxed, &b_HLTXElectronMuonRelaxed);
   fChain->SetBranchAddress("HLTXElectronTau", &HLTXElectronTau, &b_HLTXElectronTau);
   fChain->SetBranchAddress("CandHLTXElectronTauPixel", &CandHLTXElectronTauPixel, &b_CandHLTXElectronTauPixel);
   fChain->SetBranchAddress("HLTXMuonTau", &HLTXMuonTau, &b_HLTXMuonTau);
   fChain->SetBranchAddress("CandHLTEcalPi0", &CandHLTEcalPi0, &b_CandHLTEcalPi0);
   fChain->SetBranchAddress("CandHLTEcalPhiSym", &CandHLTEcalPhiSym, &b_CandHLTEcalPhiSym);
   fChain->SetBranchAddress("CandHLTHcalPhiSym", &CandHLTHcalPhiSym, &b_CandHLTHcalPhiSym);
   fChain->SetBranchAddress("HLTHcalIsolatedTrack", &HLTHcalIsolatedTrack, &b_HLTHcalIsolatedTrack);
   fChain->SetBranchAddress("CandHLTHcalIsolatedTrackNoEcalIsol", &CandHLTHcalIsolatedTrackNoEcalIsol, &b_CandHLTHcalIsolatedTrackNoEcalIsol);
   fChain->SetBranchAddress("HLTMinBiasPixel", &HLTMinBiasPixel, &b_HLTMinBiasPixel);
   fChain->SetBranchAddress("CandHLTMinBiasForAlignment", &CandHLTMinBiasForAlignment, &b_CandHLTMinBiasForAlignment);
   fChain->SetBranchAddress("HLTMinBias", &HLTMinBias, &b_HLTMinBias);
   fChain->SetBranchAddress("HLTZeroBias", &HLTZeroBias, &b_HLTZeroBias);
   fChain->SetBranchAddress("HLTriggerType", &HLTriggerType, &b_HLTriggerType);
   fChain->SetBranchAddress("TriggerFinalPath", &TriggerFinalPath, &b_TriggerFinalPath);
   fChain->SetBranchAddress("L1_DoubleEG10", &L1_DoubleEG10, &b_L1_DoubleEG10);
   fChain->SetBranchAddress("L1_DoubleEG10_ETM20", &L1_DoubleEG10_ETM20, &b_L1_DoubleEG10_ETM20);
   fChain->SetBranchAddress("L1_DoubleEG10_HTT200", &L1_DoubleEG10_HTT200, &b_L1_DoubleEG10_HTT200);
   fChain->SetBranchAddress("L1_DoubleEG10_Mu3", &L1_DoubleEG10_Mu3, &b_L1_DoubleEG10_Mu3);
   fChain->SetBranchAddress("L1_DoubleEG15", &L1_DoubleEG15, &b_L1_DoubleEG15);
   fChain->SetBranchAddress("L1_DoubleEG5", &L1_DoubleEG5, &b_L1_DoubleEG5);
   fChain->SetBranchAddress("L1_DoubleIsoEG10", &L1_DoubleIsoEG10, &b_L1_DoubleIsoEG10);
   fChain->SetBranchAddress("L1_DoubleIsoEG5_ETM20", &L1_DoubleIsoEG5_ETM20, &b_L1_DoubleIsoEG5_ETM20);
   fChain->SetBranchAddress("L1_DoubleIsoEG5_HTT200", &L1_DoubleIsoEG5_HTT200, &b_L1_DoubleIsoEG5_HTT200);
   fChain->SetBranchAddress("L1_DoubleIsoEG5_Mu3", &L1_DoubleIsoEG5_Mu3, &b_L1_DoubleIsoEG5_Mu3);
   fChain->SetBranchAddress("L1_DoubleIsoEG8", &L1_DoubleIsoEG8, &b_L1_DoubleIsoEG8);
   fChain->SetBranchAddress("L1_DoubleJet100", &L1_DoubleJet100, &b_L1_DoubleJet100);
   fChain->SetBranchAddress("L1_DoubleJet50_ETM20", &L1_DoubleJet50_ETM20, &b_L1_DoubleJet50_ETM20);
   fChain->SetBranchAddress("L1_DoubleJet50_HTT200", &L1_DoubleJet50_HTT200, &b_L1_DoubleJet50_HTT200);
   fChain->SetBranchAddress("L1_DoubleJet70", &L1_DoubleJet70, &b_L1_DoubleJet70);
   fChain->SetBranchAddress("L1_DoubleMu3", &L1_DoubleMu3, &b_L1_DoubleMu3);
   fChain->SetBranchAddress("L1_DoubleMu3_EG10", &L1_DoubleMu3_EG10, &b_L1_DoubleMu3_EG10);
   fChain->SetBranchAddress("L1_DoubleMu3_ETM20", &L1_DoubleMu3_ETM20, &b_L1_DoubleMu3_ETM20);
   fChain->SetBranchAddress("L1_DoubleMu3_HTT200", &L1_DoubleMu3_HTT200, &b_L1_DoubleMu3_HTT200);
   fChain->SetBranchAddress("L1_DoubleMu3_IsoEG5", &L1_DoubleMu3_IsoEG5, &b_L1_DoubleMu3_IsoEG5);
   fChain->SetBranchAddress("L1_DoubleTauJet20", &L1_DoubleTauJet20, &b_L1_DoubleTauJet20);
   fChain->SetBranchAddress("L1_DoubleTauJet30", &L1_DoubleTauJet30, &b_L1_DoubleTauJet30);
   fChain->SetBranchAddress("L1_DoubleTauJet35", &L1_DoubleTauJet35, &b_L1_DoubleTauJet35);
   fChain->SetBranchAddress("L1_DoubleTauJet40", &L1_DoubleTauJet40, &b_L1_DoubleTauJet40);
   fChain->SetBranchAddress("L1_DoubleTauJet40_ETM20", &L1_DoubleTauJet40_ETM20, &b_L1_DoubleTauJet40_ETM20);
   fChain->SetBranchAddress("L1_DoubleTauJet40_HTT200", &L1_DoubleTauJet40_HTT200, &b_L1_DoubleTauJet40_HTT200);
   fChain->SetBranchAddress("L1_EG10_Jet15", &L1_EG10_Jet15, &b_L1_EG10_Jet15);
   fChain->SetBranchAddress("L1_EG12_ETM30", &L1_EG12_ETM30, &b_L1_EG12_ETM30);
   fChain->SetBranchAddress("L1_EG12_HTT200", &L1_EG12_HTT200, &b_L1_EG12_HTT200);
   fChain->SetBranchAddress("L1_EG12_Jet70", &L1_EG12_Jet70, &b_L1_EG12_Jet70);
   fChain->SetBranchAddress("L1_EG12_TauJet40", &L1_EG12_TauJet40, &b_L1_EG12_TauJet40);
   fChain->SetBranchAddress("L1_ETM10", &L1_ETM10, &b_L1_ETM10);
   fChain->SetBranchAddress("L1_ETM15", &L1_ETM15, &b_L1_ETM15);
   fChain->SetBranchAddress("L1_ETM20", &L1_ETM20, &b_L1_ETM20);
   fChain->SetBranchAddress("L1_ETM40", &L1_ETM40, &b_L1_ETM40);
   fChain->SetBranchAddress("L1_ETM45", &L1_ETM45, &b_L1_ETM45);
   fChain->SetBranchAddress("L1_ETM50", &L1_ETM50, &b_L1_ETM50);
   fChain->SetBranchAddress("L1_ETM60", &L1_ETM60, &b_L1_ETM60);
   fChain->SetBranchAddress("L1_ETT60", &L1_ETT60, &b_L1_ETT60);
   fChain->SetBranchAddress("L1_ExclusiveDoubleIsoEG6", &L1_ExclusiveDoubleIsoEG6, &b_L1_ExclusiveDoubleIsoEG6);
   fChain->SetBranchAddress("L1_ExclusiveDoubleJet60", &L1_ExclusiveDoubleJet60, &b_L1_ExclusiveDoubleJet60);
   fChain->SetBranchAddress("L1_ExclusiveJet25_Gap_Jet25", &L1_ExclusiveJet25_Gap_Jet25, &b_L1_ExclusiveJet25_Gap_Jet25);
   fChain->SetBranchAddress("L1_HTT100", &L1_HTT100, &b_L1_HTT100);
   fChain->SetBranchAddress("L1_HTT100_ETM30", &L1_HTT100_ETM30, &b_L1_HTT100_ETM30);
   fChain->SetBranchAddress("L1_HTT200", &L1_HTT200, &b_L1_HTT200);
   fChain->SetBranchAddress("L1_HTT250", &L1_HTT250, &b_L1_HTT250);
   fChain->SetBranchAddress("L1_HTT300", &L1_HTT300, &b_L1_HTT300);
   fChain->SetBranchAddress("L1_HTT400", &L1_HTT400, &b_L1_HTT400);
   fChain->SetBranchAddress("L1_HTT500", &L1_HTT500, &b_L1_HTT500);
   fChain->SetBranchAddress("L1_IsoEG10_EG10", &L1_IsoEG10_EG10, &b_L1_IsoEG10_EG10);
   fChain->SetBranchAddress("L1_IsoEG10_HTT200", &L1_IsoEG10_HTT200, &b_L1_IsoEG10_HTT200);
   fChain->SetBranchAddress("L1_IsoEG10_Jet15", &L1_IsoEG10_Jet15, &b_L1_IsoEG10_Jet15);
   fChain->SetBranchAddress("L1_IsoEG10_Jet15_ForJet10", &L1_IsoEG10_Jet15_ForJet10, &b_L1_IsoEG10_Jet15_ForJet10);
   fChain->SetBranchAddress("L1_IsoEG10_Jet20", &L1_IsoEG10_Jet20, &b_L1_IsoEG10_Jet20);
   fChain->SetBranchAddress("L1_IsoEG10_Jet30", &L1_IsoEG10_Jet30, &b_L1_IsoEG10_Jet30);
   fChain->SetBranchAddress("L1_IsoEG10_Jet70", &L1_IsoEG10_Jet70, &b_L1_IsoEG10_Jet70);
   fChain->SetBranchAddress("L1_IsoEG10_TauJet20", &L1_IsoEG10_TauJet20, &b_L1_IsoEG10_TauJet20);
   fChain->SetBranchAddress("L1_IsoEG10_TauJet30", &L1_IsoEG10_TauJet30, &b_L1_IsoEG10_TauJet30);
   fChain->SetBranchAddress("L1_Jet70_ETM40", &L1_Jet70_ETM40, &b_L1_Jet70_ETM40);
   fChain->SetBranchAddress("L1_Jet70_HTT200", &L1_Jet70_HTT200, &b_L1_Jet70_HTT200);
   fChain->SetBranchAddress("L1_Jet70_TauJet40", &L1_Jet70_TauJet40, &b_L1_Jet70_TauJet40);
   fChain->SetBranchAddress("L1_MinBias_HTT10", &L1_MinBias_HTT10, &b_L1_MinBias_HTT10);
   fChain->SetBranchAddress("L1_Mu3_EG12", &L1_Mu3_EG12, &b_L1_Mu3_EG12);
   fChain->SetBranchAddress("L1_Mu3_ETM30", &L1_Mu3_ETM30, &b_L1_Mu3_ETM30);
   fChain->SetBranchAddress("L1_Mu3_HTT200", &L1_Mu3_HTT200, &b_L1_Mu3_HTT200);
   fChain->SetBranchAddress("L1_Mu3_IsoEG5", &L1_Mu3_IsoEG5, &b_L1_Mu3_IsoEG5);
   fChain->SetBranchAddress("L1_Mu3_Jet15", &L1_Mu3_Jet15, &b_L1_Mu3_Jet15);
   fChain->SetBranchAddress("L1_Mu3_Jet70", &L1_Mu3_Jet70, &b_L1_Mu3_Jet70);
   fChain->SetBranchAddress("L1_Mu5_IsoEG10", &L1_Mu5_IsoEG10, &b_L1_Mu5_IsoEG10);
   fChain->SetBranchAddress("L1_Mu5_Jet15", &L1_Mu5_Jet15, &b_L1_Mu5_Jet15);
   fChain->SetBranchAddress("L1_Mu5_Jet20", &L1_Mu5_Jet20, &b_L1_Mu5_Jet20);
   fChain->SetBranchAddress("L1_Mu5_TauJet20", &L1_Mu5_TauJet20, &b_L1_Mu5_TauJet20);
   fChain->SetBranchAddress("L1_Mu5_TauJet30", &L1_Mu5_TauJet30, &b_L1_Mu5_TauJet30);
   fChain->SetBranchAddress("L1_QuadJet40", &L1_QuadJet40, &b_L1_QuadJet40);
   fChain->SetBranchAddress("L1_SingleEG10", &L1_SingleEG10, &b_L1_SingleEG10);
   fChain->SetBranchAddress("L1_SingleEG12", &L1_SingleEG12, &b_L1_SingleEG12);
   fChain->SetBranchAddress("L1_SingleEG15", &L1_SingleEG15, &b_L1_SingleEG15);
   fChain->SetBranchAddress("L1_SingleEG20", &L1_SingleEG20, &b_L1_SingleEG20);
   fChain->SetBranchAddress("L1_SingleEG25", &L1_SingleEG25, &b_L1_SingleEG25);
   fChain->SetBranchAddress("L1_SingleEG5", &L1_SingleEG5, &b_L1_SingleEG5);
   fChain->SetBranchAddress("L1_SingleEG8", &L1_SingleEG8, &b_L1_SingleEG8);
   fChain->SetBranchAddress("L1_SingleIsoEG10", &L1_SingleIsoEG10, &b_L1_SingleIsoEG10);
   fChain->SetBranchAddress("L1_SingleIsoEG12", &L1_SingleIsoEG12, &b_L1_SingleIsoEG12);
   fChain->SetBranchAddress("L1_SingleIsoEG15", &L1_SingleIsoEG15, &b_L1_SingleIsoEG15);
   fChain->SetBranchAddress("L1_SingleIsoEG20", &L1_SingleIsoEG20, &b_L1_SingleIsoEG20);
   fChain->SetBranchAddress("L1_SingleIsoEG25", &L1_SingleIsoEG25, &b_L1_SingleIsoEG25);
   fChain->SetBranchAddress("L1_SingleIsoEG5", &L1_SingleIsoEG5, &b_L1_SingleIsoEG5);
   fChain->SetBranchAddress("L1_SingleIsoEG8", &L1_SingleIsoEG8, &b_L1_SingleIsoEG8);
   fChain->SetBranchAddress("L1_SingleJet100", &L1_SingleJet100, &b_L1_SingleJet100);
   fChain->SetBranchAddress("L1_SingleJet15", &L1_SingleJet15, &b_L1_SingleJet15);
   fChain->SetBranchAddress("L1_SingleJet150", &L1_SingleJet150, &b_L1_SingleJet150);
   fChain->SetBranchAddress("L1_SingleJet20", &L1_SingleJet20, &b_L1_SingleJet20);
   fChain->SetBranchAddress("L1_SingleJet200", &L1_SingleJet200, &b_L1_SingleJet200);
   fChain->SetBranchAddress("L1_SingleJet30", &L1_SingleJet30, &b_L1_SingleJet30);
   fChain->SetBranchAddress("L1_SingleJet50", &L1_SingleJet50, &b_L1_SingleJet50);
   fChain->SetBranchAddress("L1_SingleJet70", &L1_SingleJet70, &b_L1_SingleJet70);
   fChain->SetBranchAddress("L1_SingleMu10", &L1_SingleMu10, &b_L1_SingleMu10);
   fChain->SetBranchAddress("L1_SingleMu14", &L1_SingleMu14, &b_L1_SingleMu14);
   fChain->SetBranchAddress("L1_SingleMu20", &L1_SingleMu20, &b_L1_SingleMu20);
   fChain->SetBranchAddress("L1_SingleMu25", &L1_SingleMu25, &b_L1_SingleMu25);
   fChain->SetBranchAddress("L1_SingleMu3", &L1_SingleMu3, &b_L1_SingleMu3);
   fChain->SetBranchAddress("L1_SingleMu5", &L1_SingleMu5, &b_L1_SingleMu5);
   fChain->SetBranchAddress("L1_SingleMu7", &L1_SingleMu7, &b_L1_SingleMu7);
   fChain->SetBranchAddress("L1_SingleTauJet10", &L1_SingleTauJet10, &b_L1_SingleTauJet10);
   fChain->SetBranchAddress("L1_SingleTauJet100", &L1_SingleTauJet100, &b_L1_SingleTauJet100);
   fChain->SetBranchAddress("L1_SingleTauJet20", &L1_SingleTauJet20, &b_L1_SingleTauJet20);
   fChain->SetBranchAddress("L1_SingleTauJet30", &L1_SingleTauJet30, &b_L1_SingleTauJet30);
   fChain->SetBranchAddress("L1_SingleTauJet40", &L1_SingleTauJet40, &b_L1_SingleTauJet40);
   fChain->SetBranchAddress("L1_SingleTauJet60", &L1_SingleTauJet60, &b_L1_SingleTauJet60);
   fChain->SetBranchAddress("L1_SingleTauJet80", &L1_SingleTauJet80, &b_L1_SingleTauJet80);
   fChain->SetBranchAddress("L1_TauJet20_ETM20", &L1_TauJet20_ETM20, &b_L1_TauJet20_ETM20);
   fChain->SetBranchAddress("L1_TauJet30_ETM30", &L1_TauJet30_ETM30, &b_L1_TauJet30_ETM30);
   fChain->SetBranchAddress("L1_TauJet30_ETM40", &L1_TauJet30_ETM40, &b_L1_TauJet30_ETM40);
   fChain->SetBranchAddress("L1_TauJet40_HTT200", &L1_TauJet40_HTT200, &b_L1_TauJet40_HTT200);
   fChain->SetBranchAddress("L1_TripleEG10", &L1_TripleEG10, &b_L1_TripleEG10);
   fChain->SetBranchAddress("L1_TripleIsoEG5", &L1_TripleIsoEG5, &b_L1_TripleIsoEG5);
   fChain->SetBranchAddress("L1_TripleJet50", &L1_TripleJet50, &b_L1_TripleJet50);
   fChain->SetBranchAddress("L1_TripleMu3", &L1_TripleMu3, &b_L1_TripleMu3);
   fChain->SetBranchAddress("L1_TripleTauJet40", &L1_TripleTauJet40, &b_L1_TripleTauJet40);
   fChain->SetBranchAddress("L1_VBF_DoubleTauHad", &L1_VBF_DoubleTauHad, &b_L1_VBF_DoubleTauHad);
   fChain->SetBranchAddress("L1_VBF_ETM50", &L1_VBF_ETM50, &b_L1_VBF_ETM50);
   fChain->SetBranchAddress("L1_VBF_ETM50_veto", &L1_VBF_ETM50_veto, &b_L1_VBF_ETM50_veto);
   fChain->SetBranchAddress("L1_VBF_IsoEG10_Tau_TauHad", &L1_VBF_IsoEG10_Tau_TauHad, &b_L1_VBF_IsoEG10_Tau_TauHad);
   fChain->SetBranchAddress("L1_VBF_IsoEG15", &L1_VBF_IsoEG15, &b_L1_VBF_IsoEG15);
   fChain->SetBranchAddress("L1_VBF_Mu10", &L1_VBF_Mu10, &b_L1_VBF_Mu10);
   fChain->SetBranchAddress("L1_VBF_Mu7_Tau_TauHad", &L1_VBF_Mu7_Tau_TauHad, &b_L1_VBF_Mu7_Tau_TauHad);
   fChain->SetBranchAddress("L1_VBF_QuadJet", &L1_VBF_QuadJet, &b_L1_VBF_QuadJet);
   fChain->SetBranchAddress("L1_ZeroBias", &L1_ZeroBias, &b_L1_ZeroBias);
   fChain->SetBranchAddress("L1_SingleMuOpen", &L1_SingleMuOpen, &b_L1_SingleMuOpen); 
   fChain->SetBranchAddress("L1_SingleMuBeamHalo", &L1_SingleMuBeamHalo, &b_L1_SingleMuBeamHalo); 
   fChain->SetBranchAddress("L1_SingleEG2", &L1_SingleEG2, &b_L1_SingleEG2); 
   fChain->SetBranchAddress("L1_DoubleEG1", &L1_DoubleEG1, &b_L1_DoubleEG1); 
   fChain->SetBranchAddress("L1_QuadJet15", &L1_QuadJet15, &b_L1_QuadJet15); 
   fChain->SetBranchAddress("L1_QuadJet30", &L1_QuadJet30, &b_L1_QuadJet30); 
   fChain->SetBranchAddress("L1_ETM30", &L1_ETM30, &b_L1_ETM30); 
   fChain->SetBranchAddress("L1_EG5_TripleJet15", &L1_EG5_TripleJet15, &b_L1_EG5_TripleJet15); 
   fChain->SetBranchAddress("L1_Mu3_TripleJet15", &L1_Mu3_TripleJet15, &b_L1_Mu3_TripleJet15); 
   fChain->SetBranchAddress("L1_SingleJetCountsHFTow", &L1_SingleJetCountsHFTow, &b_L1_SingleJetCountsHFTow); 
   fChain->SetBranchAddress("L1_DoubleJetCountsHFTow", &L1_DoubleJetCountsHFTow, &b_L1_DoubleJetCountsHFTow); 
   fChain->SetBranchAddress("L1_SingleJetCountsHFRing0Sum3", &L1_SingleJetCountsHFRing0Sum3, &b_L1_SingleJetCountsHFRing0Sum3); 
   fChain->SetBranchAddress("L1_DoubleJetCountsHFRing0Sum3", &L1_DoubleJetCountsHFRing0Sum3, &b_L1_DoubleJetCountsHFRing0Sum3); 
   fChain->SetBranchAddress("L1_SingleJetCountsHFRing0Sum6", &L1_SingleJetCountsHFRing0Sum6, &b_L1_SingleJetCountsHFRing0Sum6); 

   Notify();
}


void OHltTree::SetMapBitOfStandardHLTPath() {

  map_BitOfStandardHLTPath["HLT1jet"] = HLT1jet;
  map_BitOfStandardHLTPath["HLT2jet"] = HLT2jet;
  map_BitOfStandardHLTPath["HLT3jet"] = HLT3jet;
  map_BitOfStandardHLTPath["HLT4jet"] = HLT4jet;
  map_BitOfStandardHLTPath["HLT1MET"] = HLT1MET;
  map_BitOfStandardHLTPath["HLT2jetAco"] = HLT2jetAco;
  map_BitOfStandardHLTPath["HLT1jet1METAco"] = HLT1jet1METAco;
  map_BitOfStandardHLTPath["HLT1jet1MET"] = HLT1jet1MET;
  map_BitOfStandardHLTPath["HLT2jet1MET"] = HLT2jet1MET;
  map_BitOfStandardHLTPath["HLT3jet1MET"] = HLT3jet1MET;
  map_BitOfStandardHLTPath["HLT4jet1MET"] = HLT4jet1MET;
  map_BitOfStandardHLTPath["HLT1MET1HT"] = HLT1MET1HT;
  map_BitOfStandardHLTPath["CandHLT1SumET"] = CandHLT1SumET;
  map_BitOfStandardHLTPath["HLT1jetPE1"] = HLT1jetPE1;
  map_BitOfStandardHLTPath["HLT1jetPE3"] = HLT1jetPE3;
  map_BitOfStandardHLTPath["HLT1jetPE5"] = HLT1jetPE5;
  map_BitOfStandardHLTPath["HLT1jetPE7"] = HLT1jetPE7;
  map_BitOfStandardHLTPath["HLT1METPre1"] = HLT1METPre1;
  map_BitOfStandardHLTPath["HLT1METPre2"] = HLT1METPre2;
  map_BitOfStandardHLTPath["HLT1METPre3"] = HLT1METPre3;
  map_BitOfStandardHLTPath["HLT2jetAve30"] = HLT2jetAve30;
  map_BitOfStandardHLTPath["HLT2jetAve60"] = HLT2jetAve60;
  map_BitOfStandardHLTPath["HLT2jetAve110"] = HLT2jetAve110;
  map_BitOfStandardHLTPath["HLT2jetAve150"] = HLT2jetAve150;
  map_BitOfStandardHLTPath["HLT2jetAve200"] = HLT2jetAve200;
  map_BitOfStandardHLTPath["HLT2jetvbfMET"] = HLT2jetvbfMET;
  map_BitOfStandardHLTPath["HLTS2jet1METNV"] = HLTS2jet1METNV;
  map_BitOfStandardHLTPath["HLTS2jet1METAco"] = HLTS2jet1METAco;
  map_BitOfStandardHLTPath["HLTSjet1MET1Aco"] = HLTSjet1MET1Aco;
  map_BitOfStandardHLTPath["HLTSjet2MET1Aco"] = HLTSjet2MET1Aco;
  map_BitOfStandardHLTPath["HLTS2jetMET1Aco"] = HLTS2jetMET1Aco;
  map_BitOfStandardHLTPath["HLTJetMETRapidityGap"] = HLTJetMETRapidityGap;
  map_BitOfStandardHLTPath["HLT1Electron"] = HLT1Electron;
  map_BitOfStandardHLTPath["HLT1ElectronRelaxed"] = HLT1ElectronRelaxed;
  map_BitOfStandardHLTPath["HLT2Electron"] = HLT2Electron;
  map_BitOfStandardHLTPath["HLT2ElectronRelaxed"] = HLT2ElectronRelaxed;
  map_BitOfStandardHLTPath["HLT1Photon"] = HLT1Photon;
  map_BitOfStandardHLTPath["HLT1PhotonRelaxed"] = HLT1PhotonRelaxed;
  map_BitOfStandardHLTPath["HLT2Photon"] = HLT2Photon;
  map_BitOfStandardHLTPath["HLT2PhotonRelaxed"] = HLT2PhotonRelaxed;
  map_BitOfStandardHLTPath["HLT1EMHighEt"] = HLT1EMHighEt;
  map_BitOfStandardHLTPath["HLT1EMVeryHighEt"] = HLT1EMVeryHighEt;
  map_BitOfStandardHLTPath["HLT2ElectronZCounter"] = HLT2ElectronZCounter;
  map_BitOfStandardHLTPath["HLT2ElectronExclusive"] = HLT2ElectronExclusive;
  map_BitOfStandardHLTPath["HLT2PhotonExclusive"] = HLT2PhotonExclusive;
  map_BitOfStandardHLTPath["HLT1PhotonL1Isolated"] = HLT1PhotonL1Isolated;
  map_BitOfStandardHLTPath["CandHLT1ElectronStartup"] = CandHLT1ElectronStartup;
  map_BitOfStandardHLTPath["CandHLT1ElectronRelaxedStartup"] = CandHLT1ElectronRelaxedStartup;
  map_BitOfStandardHLTPath["CandHLT2ElectronStartup"] = CandHLT2ElectronStartup;
  map_BitOfStandardHLTPath["CandHLT2ElectronRelaxedStartup"] = CandHLT2ElectronRelaxedStartup;
  map_BitOfStandardHLTPath["HLT1MuonIso"] = HLT1MuonIso;
  map_BitOfStandardHLTPath["HLT1MuonNonIso"] = HLT1MuonNonIso;
  map_BitOfStandardHLTPath["HLT2MuonIso"] = HLT2MuonIso;
  map_BitOfStandardHLTPath["HLT2MuonNonIso"] = HLT2MuonNonIso;
  map_BitOfStandardHLTPath["HLT2MuonJPsi"] = HLT2MuonJPsi;
  map_BitOfStandardHLTPath["HLT2MuonUpsilon"] = HLT2MuonUpsilon;
  map_BitOfStandardHLTPath["HLT2MuonZ"] = HLT2MuonZ;
  map_BitOfStandardHLTPath["HLTNMuonNonIso"] = HLTNMuonNonIso;
  map_BitOfStandardHLTPath["HLT2MuonSameSign"] = HLT2MuonSameSign;
  map_BitOfStandardHLTPath["HLT1MuonPrescalePt3"] = HLT1MuonPrescalePt3;
  map_BitOfStandardHLTPath["HLT1MuonPrescalePt5"] = HLT1MuonPrescalePt5;
  map_BitOfStandardHLTPath["HLT1MuonPrescalePt7x7"] = HLT1MuonPrescalePt7x7;
  map_BitOfStandardHLTPath["HLT1MuonPrescalePt7x10"] = HLT1MuonPrescalePt7x10;
  map_BitOfStandardHLTPath["HLT1MuonLevel1"] = HLT1MuonLevel1;
  map_BitOfStandardHLTPath["CandHLT1MuonPrescaleVtx2cm"] = CandHLT1MuonPrescaleVtx2cm;
  map_BitOfStandardHLTPath["CandHLT1MuonPrescaleVtx2mm"] = CandHLT1MuonPrescaleVtx2mm;
  map_BitOfStandardHLTPath["CandHLT2MuonPrescaleVtx2cm"] = CandHLT2MuonPrescaleVtx2cm;
  map_BitOfStandardHLTPath["CandHLT2MuonPrescaleVtx2mm"] = CandHLT2MuonPrescaleVtx2mm;
  map_BitOfStandardHLTPath["HLTB1Jet"] = HLTB1Jet;
  map_BitOfStandardHLTPath["HLTB2Jet"] = HLTB2Jet;
  map_BitOfStandardHLTPath["HLTB3Jet"] = HLTB3Jet;
  map_BitOfStandardHLTPath["HLTB4Jet"] = HLTB4Jet;
  map_BitOfStandardHLTPath["HLTBHT"] = HLTBHT;
  map_BitOfStandardHLTPath["HLTB1JetMu"] = HLTB1JetMu;
  map_BitOfStandardHLTPath["HLTB2JetMu"] = HLTB2JetMu;
  map_BitOfStandardHLTPath["HLTB3JetMu"] = HLTB3JetMu;
  map_BitOfStandardHLTPath["HLTB4JetMu"] = HLTB4JetMu;
  map_BitOfStandardHLTPath["HLTBHTMu"] = HLTBHTMu;
  map_BitOfStandardHLTPath["HLTBJPsiMuMu"] = HLTBJPsiMuMu;
  map_BitOfStandardHLTPath["HLT1Tau"] = HLT1Tau;
  map_BitOfStandardHLTPath["HLT1Tau1MET"] = HLT1Tau1MET;
  map_BitOfStandardHLTPath["HLT2TauPixel"] = HLT2TauPixel;
  map_BitOfStandardHLTPath["HLTXElectronBJet"] = HLTXElectronBJet;
  map_BitOfStandardHLTPath["HLTXMuonBJet"] = HLTXMuonBJet;
  map_BitOfStandardHLTPath["HLTXMuonBJetSoftMuon"] = HLTXMuonBJetSoftMuon;
  map_BitOfStandardHLTPath["HLTXElectron1Jet"] = HLTXElectron1Jet;
  map_BitOfStandardHLTPath["HLTXElectron2Jet"] = HLTXElectron2Jet;
  map_BitOfStandardHLTPath["HLTXElectron3Jet"] = HLTXElectron3Jet;
  map_BitOfStandardHLTPath["HLTXElectron4Jet"] = HLTXElectron4Jet;
  map_BitOfStandardHLTPath["HLTXMuonJets"] = HLTXMuonJets;
  map_BitOfStandardHLTPath["CandHLTXMuonNoL2IsoJets"] = CandHLTXMuonNoL2IsoJets;
  map_BitOfStandardHLTPath["CandHLTXMuonNoIsoJets"] = CandHLTXMuonNoIsoJets;
  map_BitOfStandardHLTPath["HLTXElectronMuon"] = HLTXElectronMuon;
  map_BitOfStandardHLTPath["HLTXElectronMuonRelaxed"] = HLTXElectronMuonRelaxed;
  map_BitOfStandardHLTPath["HLTXElectronTau"] = HLTXElectronTau;
  map_BitOfStandardHLTPath["CandHLTXElectronTauPixel"] = CandHLTXElectronTauPixel;
  map_BitOfStandardHLTPath["HLTXMuonTau"] = HLTXMuonTau;
  map_BitOfStandardHLTPath["CandHLTEcalPi0"] = CandHLTEcalPi0;
  map_BitOfStandardHLTPath["CandHLTEcalPhiSym"] = CandHLTEcalPhiSym;
  map_BitOfStandardHLTPath["CandHLTHcalPhiSym"] = CandHLTHcalPhiSym;
  map_BitOfStandardHLTPath["HLTHcalIsolatedTrack"] = HLTHcalIsolatedTrack;
  map_BitOfStandardHLTPath["CandHLTHcalIsolatedTrackNoEcalIsol"] = CandHLTHcalIsolatedTrackNoEcalIsol;
  map_BitOfStandardHLTPath["HLTMinBiasPixel"] = HLTMinBiasPixel;
  map_BitOfStandardHLTPath["CandHLTMinBiasForAlignment"] = CandHLTMinBiasForAlignment;
  map_BitOfStandardHLTPath["HLTMinBias"] = HLTMinBias;
  map_BitOfStandardHLTPath["HLTZeroBias"] = HLTZeroBias;

    // For measuring L1 rates, also add L1 bits to the map!
  map_BitOfStandardHLTPath["L1_SingleMuOpen"] = L1_SingleMuOpen; 
  map_BitOfStandardHLTPath["L1_SingleMu3"] = L1_SingleMu3; 
  map_BitOfStandardHLTPath["L1_SingleMu5"] = L1_SingleMu5;             
  map_BitOfStandardHLTPath["L1_SingleMu7"] = L1_SingleMu7;             
  map_BitOfStandardHLTPath["L1_SingleMu10"] = L1_SingleMu10;            
  map_BitOfStandardHLTPath["L1_SingleMuBeamHalo"] = L1_SingleMuBeamHalo; 
  map_BitOfStandardHLTPath["L1_DoubleMu3"] = L1_DoubleMu3;             
  map_BitOfStandardHLTPath["L1_TripleMu3"] = L1_TripleMu3;             
  
  map_BitOfStandardHLTPath["L1_SingleIsoEG10"] = L1_SingleIsoEG10;       
  map_BitOfStandardHLTPath["L1_SingleIsoEG12"] = L1_SingleIsoEG12;       
  map_BitOfStandardHLTPath["L1_DoubleIsoEG8"] = L1_DoubleIsoEG8;         
  
  map_BitOfStandardHLTPath["L1_SingleEG2"] = L1_SingleEG2;            
  map_BitOfStandardHLTPath["L1_SingleEG5"] = L1_SingleEG5;             
  map_BitOfStandardHLTPath["L1_SingleEG8"] = L1_SingleEG8;             
  map_BitOfStandardHLTPath["L1_SingleEG10"] = L1_SingleEG10;            
  map_BitOfStandardHLTPath["L1_SingleEG12"] = L1_SingleEG12;           
  map_BitOfStandardHLTPath["L1_SingleEG15"] = L1_SingleEG15;          
  map_BitOfStandardHLTPath["L1_DoubleEG1"] = L1_DoubleEG1;            
  map_BitOfStandardHLTPath["L1_DoubleEG5"] = L1_DoubleEG5;             
  map_BitOfStandardHLTPath["L1_DoubleEG10"] = L1_DoubleEG10;            
  
  map_BitOfStandardHLTPath["L1_SingleJet15"] = L1_SingleJet15;  
  map_BitOfStandardHLTPath["L1_SingleJet30"] = L1_SingleJet30; 
  map_BitOfStandardHLTPath["L1_SingleJet50"] = L1_SingleJet50;  
  map_BitOfStandardHLTPath["L1_SingleJet70"] = L1_SingleJet70;  
  map_BitOfStandardHLTPath["L1_SingleJet100"] = L1_SingleJet100; 
  map_BitOfStandardHLTPath["L1_SingleJet150"] = L1_SingleJet150; 
  map_BitOfStandardHLTPath["L1_SingleJet200"] = L1_SingleJet200; 
  map_BitOfStandardHLTPath["L1_DoubleJet70"] = L1_DoubleJet70;   
  map_BitOfStandardHLTPath["L1_DoubleJet100"] = L1_DoubleJet100; 
  map_BitOfStandardHLTPath["L1_TripleJet50"] = L1_TripleJet50;   
  map_BitOfStandardHLTPath["L1_QuadJet15"] = L1_QuadJet15;  
  map_BitOfStandardHLTPath["L1_QuadJet30"] = L1_QuadJet30;   
  map_BitOfStandardHLTPath["L1_HTT200"] = L1_HTT200;         
  map_BitOfStandardHLTPath["L1_HTT300"] = L1_HTT300;         
  
  map_BitOfStandardHLTPath["L1_ETM20"] = L1_ETM20;
  map_BitOfStandardHLTPath["L1_ETM30"] = L1_ETM30;   
  map_BitOfStandardHLTPath["L1_ETM40"] = L1_ETM40;   
  map_BitOfStandardHLTPath["L1_ETM50"] = L1_ETM50;   
  
  map_BitOfStandardHLTPath["L1_ETT60"] = L1_ETT60;  
  
  map_BitOfStandardHLTPath["L1_SingleTauJet30"] = L1_SingleTauJet30;                
  map_BitOfStandardHLTPath["L1_SingleTauJet40"] = L1_SingleTauJet40;  
  map_BitOfStandardHLTPath["L1_SingleTauJet60"] = L1_SingleTauJet60;    
  map_BitOfStandardHLTPath["L1_SingleTauJet80"] = L1_SingleTauJet80;    
  map_BitOfStandardHLTPath["L1_DoubleTauJet20"] = L1_DoubleTauJet20;    
  map_BitOfStandardHLTPath["L1_DoubleTauJet40"] = L1_DoubleTauJet40;    
  
  map_BitOfStandardHLTPath["L1_IsoEG10_Jet15_ForJet10"] = L1_IsoEG10_Jet15_ForJet10;  
  map_BitOfStandardHLTPath["L1_ExclusiveDoubleIsoEG6"] = L1_ExclusiveDoubleIsoEG6;    
  map_BitOfStandardHLTPath["L1_Mu5_Jet15"] = L1_Mu5_Jet15;     
  map_BitOfStandardHLTPath["L1_IsoEG10_Jet20"] = L1_IsoEG10_Jet20;  
  map_BitOfStandardHLTPath["L1_IsoEG10_Jet30"] = L1_IsoEG10_Jet30;  
  map_BitOfStandardHLTPath["L1_Mu3_IsoEG5"] = L1_Mu3_IsoEG5;   
  map_BitOfStandardHLTPath["L1_Mu3_EG12"] = L1_Mu3_EG12;     
  map_BitOfStandardHLTPath["L1_IsoEG10_TauJet20"] = L1_IsoEG10_TauJet20;  
  map_BitOfStandardHLTPath["L1_Mu5_TauJet20"] = L1_Mu5_TauJet20;  
  map_BitOfStandardHLTPath["L1_TauJet30_ETM30"] = L1_TauJet30_ETM30;  
  map_BitOfStandardHLTPath["L1_EG5_TripleJet15"] = L1_EG5_TripleJet15; 
  map_BitOfStandardHLTPath["L1_Mu3_TripleJet15"] = L1_Mu3_TripleJet15; 
  
  map_BitOfStandardHLTPath["L1_ZeroBias"] = L1_ZeroBias;
  map_BitOfStandardHLTPath["L1_MinBias_HTT10"] = L1_MinBias_HTT10;
  map_BitOfStandardHLTPath["L1_SingleJetCountsHFTow"] = L1_SingleJetCountsHFTow; 
  map_BitOfStandardHLTPath["L1_DoubleJetCountsHFTow"] = L1_DoubleJetCountsHFTow; 
  map_BitOfStandardHLTPath["L1_SingleJetCountsHFRing0Sum3"] = L1_SingleJetCountsHFRing0Sum3;  
  map_BitOfStandardHLTPath["L1_DoubleJetCountsHFRing0Sum3"] = L1_DoubleJetCountsHFRing0Sum3;   
  map_BitOfStandardHLTPath["L1_SingleJetCountsHFRing0Sum6"] = L1_SingleJetCountsHFRing0Sum6;    
}


Bool_t OHltTree::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void OHltTree::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t OHltTree::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef OHltTree_cxx
