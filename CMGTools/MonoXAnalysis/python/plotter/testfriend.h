//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon May 18 10:48:11 2015 by ROOT version 5.34/18
// from TTree tree/treeProducerDarkMatterMonoJet
// found on file: trees/Monojet_M_10_V/treeProducerDarkMatterMonoJet/tree.root
//////////////////////////////////////////////////////////

#ifndef testfriend_h
#define testfriend_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class testfriend {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Int_t           run;
   Int_t           lumi;
   Int_t           evt;
   Int_t           isData;
   Float_t         xsec;
   Float_t         puWeight;
   Float_t         weight;
   Int_t           nTrueInt;
   Float_t         genWeight;
   Float_t         rho;
   Int_t           nVert;
   Int_t           nJet25;
   Int_t           nBJetLoose25;
   Int_t           nBJetMedium25;
   Int_t           nBJetTight25;
   Int_t           nJet30;
   Int_t           nJet30a;
   Int_t           nBJetLoose30;
   Int_t           nBJetMedium30;
   Int_t           nBJetTight30;
   Int_t           nJet40;
   Int_t           nJet40a;
   Int_t           nBJetLoose40;
   Int_t           nBJetMedium40;
   Int_t           nBJetTight40;
   Int_t           nLepGood20;
   Int_t           nLepGood15;
   Int_t           nLepGood10;
   Float_t         htJet25;
   Float_t         mhtJet25;
   Float_t         htJet40j;
   Float_t         htJet40ja;
   Float_t         htJet40;
   Float_t         htJet40a;
   Float_t         mhtJet40;
   Float_t         mhtJet40a;
   Float_t         mt2_had;
   Float_t         mt2ViaKt_had;
   Float_t         mt2_bb;
   Float_t         mt2_gen;
   Float_t         mt2;
   Float_t         gamma_mt2;
   Float_t         zll_mt2;
   Float_t         mr_had;
   Float_t         mr_bb;
   Float_t         mr_lept;
   Float_t         mr_gen;
   Float_t         mr;
   Float_t         mtr_had;
   Float_t         mtr_bb;
   Float_t         mtr_lept;
   Float_t         mtr_gen;
   Float_t         mtr;
   Float_t         r_had;
   Float_t         r_bb;
   Float_t         r_lept;
   Float_t         r_gen;
   Float_t         r;
   Float_t         alphaT;
   Float_t         apcjetmetmin;
   Float_t         mZ1;
   Float_t         mZ1SFSS;
   Float_t         minMllSFOS;
   Float_t         maxMllSFOS;
   Float_t         minMllAFOS;
   Float_t         maxMllAFOS;
   Float_t         minMllAFSS;
   Float_t         maxMllAFSS;
   Float_t         minMllAFAS;
   Float_t         maxMllAFAS;
   Float_t         m2l;
   Int_t           nBJet25;
   Int_t           nFatJet100;
   Int_t           nMuons10;
   Int_t           nElectrons10;
   Int_t           nTaus20;
   Int_t           nGammas20;
   Float_t         HLT_MonoJet;
   Float_t         Flag_EcalDeadCellTriggerPrimitiveFilter;
   Float_t         Flag_trkPOG_manystripclus53X;
   Float_t         Flag_ecalLaserCorrFilter;
   Float_t         Flag_trkPOG_toomanystripclus53X;
   Float_t         Flag_hcalLaserEventFilter;
   Float_t         Flag_trkPOG_logErrorTooManyClusters;
   Float_t         Flag_trkPOGFilters;
   Float_t         Flag_trackingFailureFilter;
   Float_t         Flag_CSCTightHaloFilter;
   Float_t         Flag_HBHENoiseFilter;
   Float_t         Flag_goodVertices;
   Float_t         Flag_METFilters;
   Float_t         Flag_eeBadScFilter;
   Float_t         met_pt;
   Float_t         met_eta;
   Float_t         met_phi;
   Float_t         met_mass;
   Float_t         met_sumEt;
   Float_t         met_genPt;
   Float_t         met_genPhi;
   Float_t         met_genEta;
   Float_t         metNoMu_pt;
   Float_t         metNoMu_eta;
   Float_t         metNoMu_phi;
   Float_t         metNoMu_mass;
   Float_t         metNoMu_sumEt;
   Float_t         metNoMu_genPt;
   Float_t         metNoMu_genPhi;
   Float_t         metNoMu_genEta;
   Float_t         metNoPU_pt;
   Float_t         metNoPU_eta;
   Float_t         metNoPU_phi;
   Float_t         metNoPU_mass;
   Int_t           nFatJet;
   Int_t           FatJet_id[6];   //[nFatJet]
   Int_t           FatJet_puId[6];   //[nFatJet]
   Float_t         FatJet_btagCSV[6];   //[nFatJet]
   Float_t         FatJet_btagCMVA[6];   //[nFatJet]
   Float_t         FatJet_rawPt[6];   //[nFatJet]
   Float_t         FatJet_mcPt[6];   //[nFatJet]
   Int_t           FatJet_mcFlavour[6];   //[nFatJet]
   Int_t           FatJet_mcMatchId[6];   //[nFatJet]
   Float_t         FatJet_pt[6];   //[nFatJet]
   Float_t         FatJet_eta[6];   //[nFatJet]
   Float_t         FatJet_phi[6];   //[nFatJet]
   Float_t         FatJet_mass[6];   //[nFatJet]
   Float_t         FatJet_prunedMass[6];   //[nFatJet]
   Float_t         FatJet_trimmedMass[6];   //[nFatJet]
   Float_t         FatJet_filteredMass[6];   //[nFatJet]
   Float_t         FatJet_tau1[6];   //[nFatJet]
   Float_t         FatJet_tau2[6];   //[nFatJet]
   Float_t         FatJet_tau3[6];   //[nFatJet]
   Float_t         FatJet_topMass[6];   //[nFatJet]
   Float_t         FatJet_minMass[6];   //[nFatJet]
   Float_t         FatJet_nSubJets[6];   //[nFatJet]
   Int_t           nGenPart;
   Int_t           GenPart_motherId[14];   //[nGenPart]
   Int_t           GenPart_grandmotherId[14];   //[nGenPart]
   Int_t           GenPart_sourceId[14];   //[nGenPart]
   Float_t         GenPart_charge[14];   //[nGenPart]
   Int_t           GenPart_status[14];   //[nGenPart]
   Int_t           GenPart_pdgId[14];   //[nGenPart]
   Float_t         GenPart_pt[14];   //[nGenPart]
   Float_t         GenPart_eta[14];   //[nGenPart]
   Float_t         GenPart_phi[14];   //[nGenPart]
   Float_t         GenPart_mass[14];   //[nGenPart]
   Int_t           GenPart_motherIndex[14];   //[nGenPart]
   Int_t           nisoTrack;
   Int_t           isoTrack_pdgId[5];   //[nisoTrack]
   Float_t         isoTrack_pt[5];   //[nisoTrack]
   Float_t         isoTrack_eta[5];   //[nisoTrack]
   Float_t         isoTrack_phi[5];   //[nisoTrack]
   Float_t         isoTrack_mass[5];   //[nisoTrack]
   Int_t           isoTrack_charge[5];   //[nisoTrack]
   Float_t         isoTrack_dz[5];   //[nisoTrack]
   Float_t         isoTrack_absIso[5];   //[nisoTrack]
   Int_t           isoTrack_mcMatchId[5];   //[nisoTrack]
   Int_t           nLepGood;
   Int_t           LepGood_eleMVAId[3];   //[nLepGood]
   Float_t         LepGood_mvaId[3];   //[nLepGood]
   Float_t         LepGood_mvaIdTrig[3];   //[nLepGood]
   Float_t         LepGood_mvaIdPhys14[3];   //[nLepGood]
   Float_t         LepGood_mvaTTH[3];   //[nLepGood]
   Float_t         LepGood_mvaSusy[3];   //[nLepGood]
   Float_t         LepGood_jetPtRatio[3];   //[nLepGood]
   Float_t         LepGood_jetPtRel[3];   //[nLepGood]
   Float_t         LepGood_jetBTagCSV[3];   //[nLepGood]
   Float_t         LepGood_jetBTagCMVA[3];   //[nLepGood]
   Float_t         LepGood_jetDR[3];   //[nLepGood]
   Int_t           LepGood_charge[3];   //[nLepGood]
   Int_t           LepGood_tightId[3];   //[nLepGood]
   Int_t           LepGood_eleCutIdCSA14_25ns_v1[3];   //[nLepGood]
   Int_t           LepGood_eleCutIdCSA14_50ns_v1[3];   //[nLepGood]
   Float_t         LepGood_dxy[3];   //[nLepGood]
   Float_t         LepGood_dz[3];   //[nLepGood]
   Float_t         LepGood_edxy[3];   //[nLepGood]
   Float_t         LepGood_edz[3];   //[nLepGood]
   Float_t         LepGood_ip3d[3];   //[nLepGood]
   Float_t         LepGood_sip3d[3];   //[nLepGood]
   Int_t           LepGood_convVeto[3];   //[nLepGood]
   Int_t           LepGood_lostHits[3];   //[nLepGood]
   Float_t         LepGood_relIso03[3];   //[nLepGood]
   Float_t         LepGood_relIso04[3];   //[nLepGood]
   Float_t         LepGood_miniRelIso[3];   //[nLepGood]
   Int_t           LepGood_tightCharge[3];   //[nLepGood]
   Int_t           LepGood_mcMatchId[3];   //[nLepGood]
   Int_t           LepGood_mcMatchAny[3];   //[nLepGood]
   Int_t           LepGood_mcMatchTau[3];   //[nLepGood]
   Float_t         LepGood_mcPt[3];   //[nLepGood]
   Int_t           LepGood_mediumMuonId[3];   //[nLepGood]
   Int_t           LepGood_pdgId[3];   //[nLepGood]
   Float_t         LepGood_pt[3];   //[nLepGood]
   Float_t         LepGood_eta[3];   //[nLepGood]
   Float_t         LepGood_phi[3];   //[nLepGood]
   Float_t         LepGood_mass[3];   //[nLepGood]
   Float_t         LepGood_chargedHadRelIso03[3];   //[nLepGood]
   Float_t         LepGood_chargedHadRelIso04[3];   //[nLepGood]
   Int_t           LepGood_softMuonId[3];   //[nLepGood]
   Int_t           LepGood_pfMuonId[3];   //[nLepGood]
   Int_t           LepGood_eleCutId2012_full5x5[3];   //[nLepGood]
   Int_t           LepGood_trackerLayers[3];   //[nLepGood]
   Int_t           LepGood_pixelLayers[3];   //[nLepGood]
   Int_t           LepGood_trackerHits[3];   //[nLepGood]
   Int_t           LepGood_lostOuterHits[3];   //[nLepGood]
   Float_t         LepGood_innerTrackValidHitFraction[3];   //[nLepGood]
   Float_t         LepGood_innerTrackChi2[3];   //[nLepGood]
   Float_t         LepGood_nStations[3];   //[nLepGood]
   Float_t         LepGood_caloCompatibility[3];   //[nLepGood]
   Float_t         LepGood_globalTrackChi2[3];   //[nLepGood]
   Float_t         LepGood_trkKink[3];   //[nLepGood]
   Float_t         LepGood_segmentCompatibility[3];   //[nLepGood]
   Float_t         LepGood_chi2LocalPosition[3];   //[nLepGood]
   Float_t         LepGood_chi2LocalMomentum[3];   //[nLepGood]
   Float_t         LepGood_glbTrackProbability[3];   //[nLepGood]
   Float_t         LepGood_sigmaIEtaIEta[3];   //[nLepGood]
   Float_t         LepGood_dEtaScTrkIn[3];   //[nLepGood]
   Float_t         LepGood_dPhiScTrkIn[3];   //[nLepGood]
   Float_t         LepGood_hadronicOverEm[3];   //[nLepGood]
   Float_t         LepGood_eInvMinusPInv[3];   //[nLepGood]
   Float_t         LepGood_etaSc[3];   //[nLepGood]
   Float_t         LepGood_miniRelIsoCharged[3];   //[nLepGood]
   Float_t         LepGood_miniRelIsoNeutral[3];   //[nLepGood]
   Int_t           LepGood_hasSV[3];   //[nLepGood]
   Float_t         LepGood_svRedPt[3];   //[nLepGood]
   Float_t         LepGood_svRedM[3];   //[nLepGood]
   Float_t         LepGood_svLepSip3d[3];   //[nLepGood]
   Float_t         LepGood_svSip3d[3];   //[nLepGood]
   Float_t         LepGood_svNTracks[3];   //[nLepGood]
   Float_t         LepGood_svChi2n[3];   //[nLepGood]
   Float_t         LepGood_svDxy[3];   //[nLepGood]
   Float_t         LepGood_svMass[3];   //[nLepGood]
   Float_t         LepGood_svPt[3];   //[nLepGood]
   Float_t         LepGood_svMCMatchFraction[3];   //[nLepGood]
   Float_t         LepGood_svMva[3];   //[nLepGood]
   Float_t         LepGood_jetNDau[3];   //[nLepGood]
   Float_t         LepGood_jetNDauCharged[3];   //[nLepGood]
   Float_t         LepGood_jetNDauPV[3];   //[nLepGood]
   Float_t         LepGood_jetNDauNotPV[3];   //[nLepGood]
   Float_t         LepGood_jetmaxSignedSip3D[3];   //[nLepGood]
   Float_t         LepGood_jetmaxSip3D[3];   //[nLepGood]
   Float_t         LepGood_jetmaxSignedSip2D[3];   //[nLepGood]
   Float_t         LepGood_jetmaxSip2D[3];   //[nLepGood]
   Float_t         LepGood_jetPtRelv0[3];   //[nLepGood]
   Float_t         LepGood_jetMass[3];   //[nLepGood]
   Float_t         LepGood_jetPrunedMass[3];   //[nLepGood]
   Float_t         LepGood_jetDecDR[3];   //[nLepGood]
   Float_t         LepGood_jetDecPtRel[3];   //[nLepGood]
   Float_t         LepGood_jetDecPtRatio[3];   //[nLepGood]
   Float_t         LepGood_jetDecPrunedMass[3];   //[nLepGood]
   Float_t         LepGood_jetDecPrunedPtRatio[3];   //[nLepGood]
   Float_t         LepGood_jetDec02DR[3];   //[nLepGood]
   Float_t         LepGood_jetDec02PtRel[3];   //[nLepGood]
   Float_t         LepGood_jetDec02PtRatio[3];   //[nLepGood]
   Float_t         LepGood_jetDec02PrunedPtRatio[3];   //[nLepGood]
   Float_t         LepGood_jetDec02PrunedMass[3];   //[nLepGood]
   Int_t           ngenJet;
   Int_t           genJet_pdgId[10];   //[ngenJet]
   Float_t         genJet_pt[10];   //[ngenJet]
   Float_t         genJet_eta[10];   //[ngenJet]
   Float_t         genJet_phi[10];   //[ngenJet]
   Float_t         genJet_mass[10];   //[ngenJet]
   Float_t         genJet_charge[10];   //[ngenJet]
   Int_t           genJet_status[10];   //[ngenJet]
   Int_t           nJetFwd;
   Int_t           JetFwd_mcMatchFlav[5];   //[nJetFwd]
   Float_t         JetFwd_area[5];   //[nJetFwd]
   Float_t         JetFwd_qgl[5];   //[nJetFwd]
   Float_t         JetFwd_ptd[5];   //[nJetFwd]
   Float_t         JetFwd_axis2[5];   //[nJetFwd]
   Int_t           JetFwd_mult[5];   //[nJetFwd]
   Int_t           JetFwd_partonId[5];   //[nJetFwd]
   Int_t           JetFwd_partonMotherId[5];   //[nJetFwd]
   Float_t         JetFwd_nLeptons[5];   //[nJetFwd]
   Int_t           JetFwd_id[5];   //[nJetFwd]
   Int_t           JetFwd_puId[5];   //[nJetFwd]
   Float_t         JetFwd_btagCSV[5];   //[nJetFwd]
   Float_t         JetFwd_btagCMVA[5];   //[nJetFwd]
   Float_t         JetFwd_rawPt[5];   //[nJetFwd]
   Float_t         JetFwd_mcPt[5];   //[nJetFwd]
   Int_t           JetFwd_mcFlavour[5];   //[nJetFwd]
   Int_t           JetFwd_mcMatchId[5];   //[nJetFwd]
   Float_t         JetFwd_pt[5];   //[nJetFwd]
   Float_t         JetFwd_eta[5];   //[nJetFwd]
   Float_t         JetFwd_phi[5];   //[nJetFwd]
   Float_t         JetFwd_mass[5];   //[nJetFwd]
   Float_t         JetFwd_prunedMass[5];   //[nJetFwd]
   Int_t           JetFwd_mcNumPartons[5];   //[nJetFwd]
   Int_t           JetFwd_mcNumLeptons[5];   //[nJetFwd]
   Int_t           JetFwd_mcNumTaus[5];   //[nJetFwd]
   Float_t         JetFwd_mcAnyPartonMass[5];   //[nJetFwd]
   Int_t           JetFwd_nSubJets[5];   //[nJetFwd]
   Int_t           JetFwd_nSubJets25[5];   //[nJetFwd]
   Int_t           JetFwd_nSubJets30[5];   //[nJetFwd]
   Int_t           JetFwd_nSubJets40[5];   //[nJetFwd]
   Int_t           JetFwd_nSubJetsZ01[5];   //[nJetFwd]
   Float_t         JetFwd_chHEF[5];   //[nJetFwd]
   Float_t         JetFwd_neHEF[5];   //[nJetFwd]
   Float_t         JetFwd_phEF[5];   //[nJetFwd]
   Float_t         JetFwd_eEF[5];   //[nJetFwd]
   Float_t         JetFwd_muEF[5];   //[nJetFwd]
   Float_t         JetFwd_HFHEF[5];   //[nJetFwd]
   Float_t         JetFwd_HFEMEF[5];   //[nJetFwd]
   Int_t           JetFwd_chHMult[5];   //[nJetFwd]
   Int_t           JetFwd_neHMult[5];   //[nJetFwd]
   Int_t           JetFwd_phMult[5];   //[nJetFwd]
   Int_t           JetFwd_eMult[5];   //[nJetFwd]
   Int_t           JetFwd_muMult[5];   //[nJetFwd]
   Int_t           JetFwd_HFHMult[5];   //[nJetFwd]
   Int_t           JetFwd_HFEMMult[5];   //[nJetFwd]
   Int_t           nGammaGood;
   Int_t           GammaGood_idCutBased[1];   //[nGammaGood]
   Float_t         GammaGood_hOverE[1];   //[nGammaGood]
   Float_t         GammaGood_r9[1];   //[nGammaGood]
   Float_t         GammaGood_sigmaIetaIeta[1];   //[nGammaGood]
   Float_t         GammaGood_chHadIso[1];   //[nGammaGood]
   Float_t         GammaGood_chHadIso04[1];   //[nGammaGood]
   Float_t         GammaGood_neuHadIso[1];   //[nGammaGood]
   Float_t         GammaGood_phIso[1];   //[nGammaGood]
   Int_t           GammaGood_mcMatchId[1];   //[nGammaGood]
   Float_t         GammaGood_mcPt[1];   //[nGammaGood]
   Int_t           GammaGood_pdgId[1];   //[nGammaGood]
   Float_t         GammaGood_pt[1];   //[nGammaGood]
   Float_t         GammaGood_eta[1];   //[nGammaGood]
   Float_t         GammaGood_phi[1];   //[nGammaGood]
   Float_t         GammaGood_mass[1];   //[nGammaGood]
   Float_t         GammaGood_genIso04[1];   //[nGammaGood]
   Float_t         GammaGood_genIso[1];   //[nGammaGood]
   Float_t         GammaGood_chHadIsoRC04[1];   //[nGammaGood]
   Float_t         GammaGood_chHadIsoRC[1];   //[nGammaGood]
   Int_t           nTauGood;
   Int_t           TauGood_charge[3];   //[nTauGood]
   Int_t           TauGood_decayMode[3];   //[nTauGood]
   Int_t           TauGood_idDecayMode[3];   //[nTauGood]
   Int_t           TauGood_idDecayModeNewDMs[3];   //[nTauGood]
   Float_t         TauGood_dxy[3];   //[nTauGood]
   Float_t         TauGood_dz[3];   //[nTauGood]
   Int_t           TauGood_idMVA[3];   //[nTauGood]
   Int_t           TauGood_idMVANewDM[3];   //[nTauGood]
   Int_t           TauGood_idCI3hit[3];   //[nTauGood]
   Int_t           TauGood_idAntiMu[3];   //[nTauGood]
   Int_t           TauGood_idAntiE[3];   //[nTauGood]
   Float_t         TauGood_isoCI3hit[3];   //[nTauGood]
   Int_t           TauGood_mcMatchId[3];   //[nTauGood]
   Int_t           TauGood_pdgId[3];   //[nTauGood]
   Float_t         TauGood_pt[3];   //[nTauGood]
   Float_t         TauGood_eta[3];   //[nTauGood]
   Float_t         TauGood_phi[3];   //[nTauGood]
   Float_t         TauGood_mass[3];   //[nTauGood]
   Int_t           ngenLep;
   Int_t           genLep_motherId[1];   //[ngenLep]
   Int_t           genLep_grandmotherId[1];   //[ngenLep]
   Int_t           genLep_sourceId[1];   //[ngenLep]
   Float_t         genLep_charge[1];   //[ngenLep]
   Int_t           genLep_status[1];   //[ngenLep]
   Int_t           genLep_pdgId[1];   //[ngenLep]
   Float_t         genLep_pt[1];   //[ngenLep]
   Float_t         genLep_eta[1];   //[ngenLep]
   Float_t         genLep_phi[1];   //[ngenLep]
   Float_t         genLep_mass[1];   //[ngenLep]
   Int_t           genLep_motherIndex[1];   //[ngenLep]
   Int_t           nLHEweight;
   Int_t           LHEweight_id[1];   //[nLHEweight]
   Float_t         LHEweight_wgt[1];   //[nLHEweight]
   Int_t           nJet;
   Int_t           Jet_mcMatchFlav[10];   //[nJet]
   Float_t         Jet_area[10];   //[nJet]
   Float_t         Jet_qgl[10];   //[nJet]
   Float_t         Jet_ptd[10];   //[nJet]
   Float_t         Jet_axis2[10];   //[nJet]
   Int_t           Jet_mult[10];   //[nJet]
   Int_t           Jet_partonId[10];   //[nJet]
   Int_t           Jet_partonMotherId[10];   //[nJet]
   Float_t         Jet_nLeptons[10];   //[nJet]
   Int_t           Jet_id[10];   //[nJet]
   Int_t           Jet_puId[10];   //[nJet]
   Float_t         Jet_btagCSV[10];   //[nJet]
   Float_t         Jet_btagCMVA[10];   //[nJet]
   Float_t         Jet_rawPt[10];   //[nJet]
   Float_t         Jet_mcPt[10];   //[nJet]
   Int_t           Jet_mcFlavour[10];   //[nJet]
   Int_t           Jet_mcMatchId[10];   //[nJet]
   Float_t         Jet_pt[10];   //[nJet]
   Float_t         Jet_eta[10];   //[nJet]
   Float_t         Jet_phi[10];   //[nJet]
   Float_t         Jet_mass[10];   //[nJet]
   Float_t         Jet_prunedMass[10];   //[nJet]
   Int_t           Jet_mcNumPartons[10];   //[nJet]
   Int_t           Jet_mcNumLeptons[10];   //[nJet]
   Int_t           Jet_mcNumTaus[10];   //[nJet]
   Float_t         Jet_mcAnyPartonMass[10];   //[nJet]
   Int_t           Jet_nSubJets[10];   //[nJet]
   Int_t           Jet_nSubJets25[10];   //[nJet]
   Int_t           Jet_nSubJets30[10];   //[nJet]
   Int_t           Jet_nSubJets40[10];   //[nJet]
   Int_t           Jet_nSubJetsZ01[10];   //[nJet]
   Float_t         Jet_chHEF[10];   //[nJet]
   Float_t         Jet_neHEF[10];   //[nJet]
   Float_t         Jet_phEF[10];   //[nJet]
   Float_t         Jet_eEF[10];   //[nJet]
   Float_t         Jet_muEF[10];   //[nJet]
   Float_t         Jet_HFHEF[10];   //[nJet]
   Float_t         Jet_HFEMEF[10];   //[nJet]
   Int_t           Jet_chHMult[10];   //[nJet]
   Int_t           Jet_neHMult[10];   //[nJet]
   Int_t           Jet_phMult[10];   //[nJet]
   Int_t           Jet_eMult[10];   //[nJet]
   Int_t           Jet_muMult[10];   //[nJet]
   Int_t           Jet_HFHMult[10];   //[nJet]
   Int_t           Jet_HFEMMult[10];   //[nJet]

   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_lumi;   //!
   TBranch        *b_evt;   //!
   TBranch        *b_isData;   //!
   TBranch        *b_xsec;   //!
   TBranch        *b_puWeight;   //!
   TBranch        *b_nTrueInt;   //!
   TBranch        *b_genWeight;   //!
   TBranch        *b_rho;   //!
   TBranch        *b_nVert;   //!
   TBranch        *b_nJet25;   //!
   TBranch        *b_nBJetLoose25;   //!
   TBranch        *b_nBJetMedium25;   //!
   TBranch        *b_nBJetTight25;   //!
   TBranch        *b_nJet30;   //!
   TBranch        *b_nJet30a;   //!
   TBranch        *b_nBJetLoose30;   //!
   TBranch        *b_nBJetMedium30;   //!
   TBranch        *b_nBJetTight30;   //!
   TBranch        *b_nJet40;   //!
   TBranch        *b_nJet40a;   //!
   TBranch        *b_nBJetLoose40;   //!
   TBranch        *b_nBJetMedium40;   //!
   TBranch        *b_nBJetTight40;   //!
   TBranch        *b_nLepGood20;   //!
   TBranch        *b_nLepGood15;   //!
   TBranch        *b_nLepGood10;   //!
   TBranch        *b_htJet25;   //!
   TBranch        *b_mhtJet25;   //!
   TBranch        *b_htJet40j;   //!
   TBranch        *b_htJet40ja;   //!
   TBranch        *b_htJet40;   //!
   TBranch        *b_htJet40a;   //!
   TBranch        *b_mhtJet40;   //!
   TBranch        *b_mhtJet40a;   //!
   TBranch        *b_mt2_had;   //!
   TBranch        *b_mt2ViaKt_had;   //!
   TBranch        *b_mt2_bb;   //!
   TBranch        *b_mt2_gen;   //!
   TBranch        *b_mt2;   //!
   TBranch        *b_gamma_mt2;   //!
   TBranch        *b_zll_mt2;   //!
   TBranch        *b_mr_had;   //!
   TBranch        *b_mr_bb;   //!
   TBranch        *b_mr_lept;   //!
   TBranch        *b_mr_gen;   //!
   TBranch        *b_mr;   //!
   TBranch        *b_mtr_had;   //!
   TBranch        *b_mtr_bb;   //!
   TBranch        *b_mtr_lept;   //!
   TBranch        *b_mtr_gen;   //!
   TBranch        *b_mtr;   //!
   TBranch        *b_r_had;   //!
   TBranch        *b_r_bb;   //!
   TBranch        *b_r_lept;   //!
   TBranch        *b_r_gen;   //!
   TBranch        *b_r;   //!
   TBranch        *b_alphaT;   //!
   TBranch        *b_apcjetmetmin;   //!
   TBranch        *b_mZ1;   //!
   TBranch        *b_mZ1SFSS;   //!
   TBranch        *b_minMllSFOS;   //!
   TBranch        *b_maxMllSFOS;   //!
   TBranch        *b_minMllAFOS;   //!
   TBranch        *b_maxMllAFOS;   //!
   TBranch        *b_minMllAFSS;   //!
   TBranch        *b_maxMllAFSS;   //!
   TBranch        *b_minMllAFAS;   //!
   TBranch        *b_maxMllAFAS;   //!
   TBranch        *b_m2l;   //!
   TBranch        *b_nBJet25;   //!
   TBranch        *b_nFatJet100;   //!
   TBranch        *b_nMuons10;   //!
   TBranch        *b_nElectrons10;   //!
   TBranch        *b_nTaus20;   //!
   TBranch        *b_nGammas20;   //!
   TBranch        *b_HLT_MonoJet;   //!
   TBranch        *b_Flag_EcalDeadCellTriggerPrimitiveFilter;   //!
   TBranch        *b_Flag_trkPOG_manystripclus53X;   //!
   TBranch        *b_Flag_ecalLaserCorrFilter;   //!
   TBranch        *b_Flag_trkPOG_toomanystripclus53X;   //!
   TBranch        *b_Flag_hcalLaserEventFilter;   //!
   TBranch        *b_Flag_trkPOG_logErrorTooManyClusters;   //!
   TBranch        *b_Flag_trkPOGFilters;   //!
   TBranch        *b_Flag_trackingFailureFilter;   //!
   TBranch        *b_Flag_CSCTightHaloFilter;   //!
   TBranch        *b_Flag_HBHENoiseFilter;   //!
   TBranch        *b_Flag_goodVertices;   //!
   TBranch        *b_Flag_METFilters;   //!
   TBranch        *b_Flag_eeBadScFilter;   //!
   TBranch        *b_met_pt;   //!
   TBranch        *b_met_eta;   //!
   TBranch        *b_met_phi;   //!
   TBranch        *b_met_mass;   //!
   TBranch        *b_met_sumEt;   //!
   TBranch        *b_met_genPt;   //!
   TBranch        *b_met_genPhi;   //!
   TBranch        *b_met_genEta;   //!
   TBranch        *b_metNoMu_pt;   //!
   TBranch        *b_metNoMu_eta;   //!
   TBranch        *b_metNoMu_phi;   //!
   TBranch        *b_metNoMu_mass;   //!
   TBranch        *b_metNoMu_sumEt;   //!
   TBranch        *b_metNoMu_genPt;   //!
   TBranch        *b_metNoMu_genPhi;   //!
   TBranch        *b_metNoMu_genEta;   //!
   TBranch        *b_metNoPU_pt;   //!
   TBranch        *b_metNoPU_eta;   //!
   TBranch        *b_metNoPU_phi;   //!
   TBranch        *b_metNoPU_mass;   //!
   TBranch        *b_nFatJet;   //!
   TBranch        *b_FatJet_id;   //!
   TBranch        *b_FatJet_puId;   //!
   TBranch        *b_FatJet_btagCSV;   //!
   TBranch        *b_FatJet_btagCMVA;   //!
   TBranch        *b_FatJet_rawPt;   //!
   TBranch        *b_FatJet_mcPt;   //!
   TBranch        *b_FatJet_mcFlavour;   //!
   TBranch        *b_FatJet_mcMatchId;   //!
   TBranch        *b_FatJet_pt;   //!
   TBranch        *b_FatJet_eta;   //!
   TBranch        *b_FatJet_phi;   //!
   TBranch        *b_FatJet_mass;   //!
   TBranch        *b_FatJet_prunedMass;   //!
   TBranch        *b_FatJet_trimmedMass;   //!
   TBranch        *b_FatJet_filteredMass;   //!
   TBranch        *b_FatJet_tau1;   //!
   TBranch        *b_FatJet_tau2;   //!
   TBranch        *b_FatJet_tau3;   //!
   TBranch        *b_FatJet_topMass;   //!
   TBranch        *b_FatJet_minMass;   //!
   TBranch        *b_FatJet_nSubJets;   //!
   TBranch        *b_nGenPart;   //!
   TBranch        *b_GenPart_motherId;   //!
   TBranch        *b_GenPart_grandmotherId;   //!
   TBranch        *b_GenPart_sourceId;   //!
   TBranch        *b_GenPart_charge;   //!
   TBranch        *b_GenPart_status;   //!
   TBranch        *b_GenPart_pdgId;   //!
   TBranch        *b_GenPart_pt;   //!
   TBranch        *b_GenPart_eta;   //!
   TBranch        *b_GenPart_phi;   //!
   TBranch        *b_GenPart_mass;   //!
   TBranch        *b_GenPart_motherIndex;   //!
   TBranch        *b_nisoTrack;   //!
   TBranch        *b_isoTrack_pdgId;   //!
   TBranch        *b_isoTrack_pt;   //!
   TBranch        *b_isoTrack_eta;   //!
   TBranch        *b_isoTrack_phi;   //!
   TBranch        *b_isoTrack_mass;   //!
   TBranch        *b_isoTrack_charge;   //!
   TBranch        *b_isoTrack_dz;   //!
   TBranch        *b_isoTrack_absIso;   //!
   TBranch        *b_isoTrack_mcMatchId;   //!
   TBranch        *b_nLepGood;   //!
   TBranch        *b_LepGood_eleMVAId;   //!
   TBranch        *b_LepGood_mvaId;   //!
   TBranch        *b_LepGood_mvaIdTrig;   //!
   TBranch        *b_LepGood_mvaIdPhys14;   //!
   TBranch        *b_LepGood_mvaTTH;   //!
   TBranch        *b_LepGood_mvaSusy;   //!
   TBranch        *b_LepGood_jetPtRatio;   //!
   TBranch        *b_LepGood_jetPtRel;   //!
   TBranch        *b_LepGood_jetBTagCSV;   //!
   TBranch        *b_LepGood_jetBTagCMVA;   //!
   TBranch        *b_LepGood_jetDR;   //!
   TBranch        *b_LepGood_charge;   //!
   TBranch        *b_LepGood_tightId;   //!
   TBranch        *b_LepGood_eleCutIdCSA14_25ns_v1;   //!
   TBranch        *b_LepGood_eleCutIdCSA14_50ns_v1;   //!
   TBranch        *b_LepGood_dxy;   //!
   TBranch        *b_LepGood_dz;   //!
   TBranch        *b_LepGood_edxy;   //!
   TBranch        *b_LepGood_edz;   //!
   TBranch        *b_LepGood_ip3d;   //!
   TBranch        *b_LepGood_sip3d;   //!
   TBranch        *b_LepGood_convVeto;   //!
   TBranch        *b_LepGood_lostHits;   //!
   TBranch        *b_LepGood_relIso03;   //!
   TBranch        *b_LepGood_relIso04;   //!
   TBranch        *b_LepGood_miniRelIso;   //!
   TBranch        *b_LepGood_tightCharge;   //!
   TBranch        *b_LepGood_mcMatchId;   //!
   TBranch        *b_LepGood_mcMatchAny;   //!
   TBranch        *b_LepGood_mcMatchTau;   //!
   TBranch        *b_LepGood_mcPt;   //!
   TBranch        *b_LepGood_mediumMuonId;   //!
   TBranch        *b_LepGood_pdgId;   //!
   TBranch        *b_LepGood_pt;   //!
   TBranch        *b_LepGood_eta;   //!
   TBranch        *b_LepGood_phi;   //!
   TBranch        *b_LepGood_mass;   //!
   TBranch        *b_LepGood_chargedHadRelIso03;   //!
   TBranch        *b_LepGood_chargedHadRelIso04;   //!
   TBranch        *b_LepGood_softMuonId;   //!
   TBranch        *b_LepGood_pfMuonId;   //!
   TBranch        *b_LepGood_eleCutId2012_full5x5;   //!
   TBranch        *b_LepGood_trackerLayers;   //!
   TBranch        *b_LepGood_pixelLayers;   //!
   TBranch        *b_LepGood_trackerHits;   //!
   TBranch        *b_LepGood_lostOuterHits;   //!
   TBranch        *b_LepGood_innerTrackValidHitFraction;   //!
   TBranch        *b_LepGood_innerTrackChi2;   //!
   TBranch        *b_LepGood_nStations;   //!
   TBranch        *b_LepGood_caloCompatibility;   //!
   TBranch        *b_LepGood_globalTrackChi2;   //!
   TBranch        *b_LepGood_trkKink;   //!
   TBranch        *b_LepGood_segmentCompatibility;   //!
   TBranch        *b_LepGood_chi2LocalPosition;   //!
   TBranch        *b_LepGood_chi2LocalMomentum;   //!
   TBranch        *b_LepGood_glbTrackProbability;   //!
   TBranch        *b_LepGood_sigmaIEtaIEta;   //!
   TBranch        *b_LepGood_dEtaScTrkIn;   //!
   TBranch        *b_LepGood_dPhiScTrkIn;   //!
   TBranch        *b_LepGood_hadronicOverEm;   //!
   TBranch        *b_LepGood_eInvMinusPInv;   //!
   TBranch        *b_LepGood_etaSc;   //!
   TBranch        *b_LepGood_miniRelIsoCharged;   //!
   TBranch        *b_LepGood_miniRelIsoNeutral;   //!
   TBranch        *b_LepGood_hasSV;   //!
   TBranch        *b_LepGood_svRedPt;   //!
   TBranch        *b_LepGood_svRedM;   //!
   TBranch        *b_LepGood_svLepSip3d;   //!
   TBranch        *b_LepGood_svSip3d;   //!
   TBranch        *b_LepGood_svNTracks;   //!
   TBranch        *b_LepGood_svChi2n;   //!
   TBranch        *b_LepGood_svDxy;   //!
   TBranch        *b_LepGood_svMass;   //!
   TBranch        *b_LepGood_svPt;   //!
   TBranch        *b_LepGood_svMCMatchFraction;   //!
   TBranch        *b_LepGood_svMva;   //!
   TBranch        *b_LepGood_jetNDau;   //!
   TBranch        *b_LepGood_jetNDauCharged;   //!
   TBranch        *b_LepGood_jetNDauPV;   //!
   TBranch        *b_LepGood_jetNDauNotPV;   //!
   TBranch        *b_LepGood_jetmaxSignedSip3D;   //!
   TBranch        *b_LepGood_jetmaxSip3D;   //!
   TBranch        *b_LepGood_jetmaxSignedSip2D;   //!
   TBranch        *b_LepGood_jetmaxSip2D;   //!
   TBranch        *b_LepGood_jetPtRelv0;   //!
   TBranch        *b_LepGood_jetMass;   //!
   TBranch        *b_LepGood_jetPrunedMass;   //!
   TBranch        *b_LepGood_jetDecDR;   //!
   TBranch        *b_LepGood_jetDecPtRel;   //!
   TBranch        *b_LepGood_jetDecPtRatio;   //!
   TBranch        *b_LepGood_jetDecPrunedMass;   //!
   TBranch        *b_LepGood_jetDecPrunedPtRatio;   //!
   TBranch        *b_LepGood_jetDec02DR;   //!
   TBranch        *b_LepGood_jetDec02PtRel;   //!
   TBranch        *b_LepGood_jetDec02PtRatio;   //!
   TBranch        *b_LepGood_jetDec02PrunedPtRatio;   //!
   TBranch        *b_LepGood_jetDec02PrunedMass;   //!
   TBranch        *b_ngenJet;   //!
   TBranch        *b_genJet_pdgId;   //!
   TBranch        *b_genJet_pt;   //!
   TBranch        *b_genJet_eta;   //!
   TBranch        *b_genJet_phi;   //!
   TBranch        *b_genJet_mass;   //!
   TBranch        *b_genJet_charge;   //!
   TBranch        *b_genJet_status;   //!
   TBranch        *b_nJetFwd;   //!
   TBranch        *b_JetFwd_mcMatchFlav;   //!
   TBranch        *b_JetFwd_area;   //!
   TBranch        *b_JetFwd_qgl;   //!
   TBranch        *b_JetFwd_ptd;   //!
   TBranch        *b_JetFwd_axis2;   //!
   TBranch        *b_JetFwd_mult;   //!
   TBranch        *b_JetFwd_partonId;   //!
   TBranch        *b_JetFwd_partonMotherId;   //!
   TBranch        *b_JetFwd_nLeptons;   //!
   TBranch        *b_JetFwd_id;   //!
   TBranch        *b_JetFwd_puId;   //!
   TBranch        *b_JetFwd_btagCSV;   //!
   TBranch        *b_JetFwd_btagCMVA;   //!
   TBranch        *b_JetFwd_rawPt;   //!
   TBranch        *b_JetFwd_mcPt;   //!
   TBranch        *b_JetFwd_mcFlavour;   //!
   TBranch        *b_JetFwd_mcMatchId;   //!
   TBranch        *b_JetFwd_pt;   //!
   TBranch        *b_JetFwd_eta;   //!
   TBranch        *b_JetFwd_phi;   //!
   TBranch        *b_JetFwd_mass;   //!
   TBranch        *b_JetFwd_prunedMass;   //!
   TBranch        *b_JetFwd_mcNumPartons;   //!
   TBranch        *b_JetFwd_mcNumLeptons;   //!
   TBranch        *b_JetFwd_mcNumTaus;   //!
   TBranch        *b_JetFwd_mcAnyPartonMass;   //!
   TBranch        *b_JetFwd_nSubJets;   //!
   TBranch        *b_JetFwd_nSubJets25;   //!
   TBranch        *b_JetFwd_nSubJets30;   //!
   TBranch        *b_JetFwd_nSubJets40;   //!
   TBranch        *b_JetFwd_nSubJetsZ01;   //!
   TBranch        *b_JetFwd_chHEF;   //!
   TBranch        *b_JetFwd_neHEF;   //!
   TBranch        *b_JetFwd_phEF;   //!
   TBranch        *b_JetFwd_eEF;   //!
   TBranch        *b_JetFwd_muEF;   //!
   TBranch        *b_JetFwd_HFHEF;   //!
   TBranch        *b_JetFwd_HFEMEF;   //!
   TBranch        *b_JetFwd_chHMult;   //!
   TBranch        *b_JetFwd_neHMult;   //!
   TBranch        *b_JetFwd_phMult;   //!
   TBranch        *b_JetFwd_eMult;   //!
   TBranch        *b_JetFwd_muMult;   //!
   TBranch        *b_JetFwd_HFHMult;   //!
   TBranch        *b_JetFwd_HFEMMult;   //!
   TBranch        *b_nGammaGood;   //!
   TBranch        *b_GammaGood_idCutBased;   //!
   TBranch        *b_GammaGood_hOverE;   //!
   TBranch        *b_GammaGood_r9;   //!
   TBranch        *b_GammaGood_sigmaIetaIeta;   //!
   TBranch        *b_GammaGood_chHadIso;   //!
   TBranch        *b_GammaGood_chHadIso04;   //!
   TBranch        *b_GammaGood_neuHadIso;   //!
   TBranch        *b_GammaGood_phIso;   //!
   TBranch        *b_GammaGood_mcMatchId;   //!
   TBranch        *b_GammaGood_mcPt;   //!
   TBranch        *b_GammaGood_pdgId;   //!
   TBranch        *b_GammaGood_pt;   //!
   TBranch        *b_GammaGood_eta;   //!
   TBranch        *b_GammaGood_phi;   //!
   TBranch        *b_GammaGood_mass;   //!
   TBranch        *b_GammaGood_genIso04;   //!
   TBranch        *b_GammaGood_genIso;   //!
   TBranch        *b_GammaGood_chHadIsoRC04;   //!
   TBranch        *b_GammaGood_chHadIsoRC;   //!
   TBranch        *b_nTauGood;   //!
   TBranch        *b_TauGood_charge;   //!
   TBranch        *b_TauGood_decayMode;   //!
   TBranch        *b_TauGood_idDecayMode;   //!
   TBranch        *b_TauGood_idDecayModeNewDMs;   //!
   TBranch        *b_TauGood_dxy;   //!
   TBranch        *b_TauGood_dz;   //!
   TBranch        *b_TauGood_idMVA;   //!
   TBranch        *b_TauGood_idMVANewDM;   //!
   TBranch        *b_TauGood_idCI3hit;   //!
   TBranch        *b_TauGood_idAntiMu;   //!
   TBranch        *b_TauGood_idAntiE;   //!
   TBranch        *b_TauGood_isoCI3hit;   //!
   TBranch        *b_TauGood_mcMatchId;   //!
   TBranch        *b_TauGood_pdgId;   //!
   TBranch        *b_TauGood_pt;   //!
   TBranch        *b_TauGood_eta;   //!
   TBranch        *b_TauGood_phi;   //!
   TBranch        *b_TauGood_mass;   //!
   TBranch        *b_ngenLep;   //!
   TBranch        *b_genLep_motherId;   //!
   TBranch        *b_genLep_grandmotherId;   //!
   TBranch        *b_genLep_sourceId;   //!
   TBranch        *b_genLep_charge;   //!
   TBranch        *b_genLep_status;   //!
   TBranch        *b_genLep_pdgId;   //!
   TBranch        *b_genLep_pt;   //!
   TBranch        *b_genLep_eta;   //!
   TBranch        *b_genLep_phi;   //!
   TBranch        *b_genLep_mass;   //!
   TBranch        *b_genLep_motherIndex;   //!
   TBranch        *b_nLHEweight;   //!
   TBranch        *b_LHEweight_id;   //!
   TBranch        *b_LHEweight_wgt;   //!
   TBranch        *b_nJet;   //!
   TBranch        *b_Jet_mcMatchFlav;   //!
   TBranch        *b_Jet_area;   //!
   TBranch        *b_Jet_qgl;   //!
   TBranch        *b_Jet_ptd;   //!
   TBranch        *b_Jet_axis2;   //!
   TBranch        *b_Jet_mult;   //!
   TBranch        *b_Jet_partonId;   //!
   TBranch        *b_Jet_partonMotherId;   //!
   TBranch        *b_Jet_nLeptons;   //!
   TBranch        *b_Jet_id;   //!
   TBranch        *b_Jet_puId;   //!
   TBranch        *b_Jet_btagCSV;   //!
   TBranch        *b_Jet_btagCMVA;   //!
   TBranch        *b_Jet_rawPt;   //!
   TBranch        *b_Jet_mcPt;   //!
   TBranch        *b_Jet_mcFlavour;   //!
   TBranch        *b_Jet_mcMatchId;   //!
   TBranch        *b_Jet_pt;   //!
   TBranch        *b_Jet_eta;   //!
   TBranch        *b_Jet_phi;   //!
   TBranch        *b_Jet_mass;   //!
   TBranch        *b_Jet_prunedMass;   //!
   TBranch        *b_Jet_mcNumPartons;   //!
   TBranch        *b_Jet_mcNumLeptons;   //!
   TBranch        *b_Jet_mcNumTaus;   //!
   TBranch        *b_Jet_mcAnyPartonMass;   //!
   TBranch        *b_Jet_nSubJets;   //!
   TBranch        *b_Jet_nSubJets25;   //!
   TBranch        *b_Jet_nSubJets30;   //!
   TBranch        *b_Jet_nSubJets40;   //!
   TBranch        *b_Jet_nSubJetsZ01;   //!
   TBranch        *b_Jet_chHEF;   //!
   TBranch        *b_Jet_neHEF;   //!
   TBranch        *b_Jet_phEF;   //!
   TBranch        *b_Jet_eEF;   //!
   TBranch        *b_Jet_muEF;   //!
   TBranch        *b_Jet_HFHEF;   //!
   TBranch        *b_Jet_HFEMEF;   //!
   TBranch        *b_Jet_chHMult;   //!
   TBranch        *b_Jet_neHMult;   //!
   TBranch        *b_Jet_phMult;   //!
   TBranch        *b_Jet_eMult;   //!
   TBranch        *b_Jet_muMult;   //!
   TBranch        *b_Jet_HFHMult;   //!
   TBranch        *b_Jet_HFEMMult;   //!

   testfriend(TTree *tree=0);
   virtual ~testfriend();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef testfriend_cxx
testfriend::testfriend(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("trees/Monojet_M_10_V/treeProducerDarkMatterMonoJet/tree.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("trees/Monojet_M_10_V/treeProducerDarkMatterMonoJet/tree.root");
      }
      f->GetObject("tree",tree);

   }
   Init(tree);
}

testfriend::~testfriend()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t testfriend::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t testfriend::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void testfriend::Init(TTree *tree)
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
   fChain->AddFriend("mjvars/t","trees/0_eventvars_mj_v1/evVarFriend_Monojet_M_10_V.root");

   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("lumi", &lumi, &b_lumi);
   fChain->SetBranchAddress("evt", &evt, &b_evt);
   fChain->SetBranchAddress("isData", &isData, &b_isData);
   fChain->SetBranchAddress("xsec", &xsec, &b_xsec);
   fChain->SetBranchAddress("puWeight", &puWeight, &b_puWeight);
   fChain->SetBranchAddress("weight", &weight);
   fChain->SetBranchAddress("nTrueInt", &nTrueInt, &b_nTrueInt);
   fChain->SetBranchAddress("genWeight", &genWeight, &b_genWeight);
   fChain->SetBranchAddress("rho", &rho, &b_rho);
   fChain->SetBranchAddress("nVert", &nVert, &b_nVert);
   fChain->SetBranchAddress("nJet25", &nJet25, &b_nJet25);
   fChain->SetBranchAddress("nBJetLoose25", &nBJetLoose25, &b_nBJetLoose25);
   fChain->SetBranchAddress("nBJetMedium25", &nBJetMedium25, &b_nBJetMedium25);
   fChain->SetBranchAddress("nBJetTight25", &nBJetTight25, &b_nBJetTight25);
   fChain->SetBranchAddress("nJet30", &nJet30, &b_nJet30);
   fChain->SetBranchAddress("nJet30a", &nJet30a, &b_nJet30a);
   fChain->SetBranchAddress("nBJetLoose30", &nBJetLoose30, &b_nBJetLoose30);
   fChain->SetBranchAddress("nBJetMedium30", &nBJetMedium30, &b_nBJetMedium30);
   fChain->SetBranchAddress("nBJetTight30", &nBJetTight30, &b_nBJetTight30);
   fChain->SetBranchAddress("nJet40", &nJet40, &b_nJet40);
   fChain->SetBranchAddress("nJet40a", &nJet40a, &b_nJet40a);
   fChain->SetBranchAddress("nBJetLoose40", &nBJetLoose40, &b_nBJetLoose40);
   fChain->SetBranchAddress("nBJetMedium40", &nBJetMedium40, &b_nBJetMedium40);
   fChain->SetBranchAddress("nBJetTight40", &nBJetTight40, &b_nBJetTight40);
   fChain->SetBranchAddress("nLepGood20", &nLepGood20, &b_nLepGood20);
   fChain->SetBranchAddress("nLepGood15", &nLepGood15, &b_nLepGood15);
   fChain->SetBranchAddress("nLepGood10", &nLepGood10, &b_nLepGood10);
   fChain->SetBranchAddress("htJet25", &htJet25, &b_htJet25);
   fChain->SetBranchAddress("mhtJet25", &mhtJet25, &b_mhtJet25);
   fChain->SetBranchAddress("htJet40j", &htJet40j, &b_htJet40j);
   fChain->SetBranchAddress("htJet40ja", &htJet40ja, &b_htJet40ja);
   fChain->SetBranchAddress("htJet40", &htJet40, &b_htJet40);
   fChain->SetBranchAddress("htJet40a", &htJet40a, &b_htJet40a);
   fChain->SetBranchAddress("mhtJet40", &mhtJet40, &b_mhtJet40);
   fChain->SetBranchAddress("mhtJet40a", &mhtJet40a, &b_mhtJet40a);
   fChain->SetBranchAddress("mt2_had", &mt2_had, &b_mt2_had);
   fChain->SetBranchAddress("mt2ViaKt_had", &mt2ViaKt_had, &b_mt2ViaKt_had);
   fChain->SetBranchAddress("mt2_bb", &mt2_bb, &b_mt2_bb);
   fChain->SetBranchAddress("mt2_gen", &mt2_gen, &b_mt2_gen);
   fChain->SetBranchAddress("mt2", &mt2, &b_mt2);
   fChain->SetBranchAddress("gamma_mt2", &gamma_mt2, &b_gamma_mt2);
   fChain->SetBranchAddress("zll_mt2", &zll_mt2, &b_zll_mt2);
   fChain->SetBranchAddress("mr_had", &mr_had, &b_mr_had);
   fChain->SetBranchAddress("mr_bb", &mr_bb, &b_mr_bb);
   fChain->SetBranchAddress("mr_lept", &mr_lept, &b_mr_lept);
   fChain->SetBranchAddress("mr_gen", &mr_gen, &b_mr_gen);
   fChain->SetBranchAddress("mr", &mr, &b_mr);
   fChain->SetBranchAddress("mtr_had", &mtr_had, &b_mtr_had);
   fChain->SetBranchAddress("mtr_bb", &mtr_bb, &b_mtr_bb);
   fChain->SetBranchAddress("mtr_lept", &mtr_lept, &b_mtr_lept);
   fChain->SetBranchAddress("mtr_gen", &mtr_gen, &b_mtr_gen);
   fChain->SetBranchAddress("mtr", &mtr, &b_mtr);
   fChain->SetBranchAddress("r_had", &r_had, &b_r_had);
   fChain->SetBranchAddress("r_bb", &r_bb, &b_r_bb);
   fChain->SetBranchAddress("r_lept", &r_lept, &b_r_lept);
   fChain->SetBranchAddress("r_gen", &r_gen, &b_r_gen);
   fChain->SetBranchAddress("r", &r, &b_r);
   fChain->SetBranchAddress("alphaT", &alphaT, &b_alphaT);
   fChain->SetBranchAddress("apcjetmetmin", &apcjetmetmin, &b_apcjetmetmin);
   fChain->SetBranchAddress("mZ1", &mZ1, &b_mZ1);
   fChain->SetBranchAddress("mZ1SFSS", &mZ1SFSS, &b_mZ1SFSS);
   fChain->SetBranchAddress("minMllSFOS", &minMllSFOS, &b_minMllSFOS);
   fChain->SetBranchAddress("maxMllSFOS", &maxMllSFOS, &b_maxMllSFOS);
   fChain->SetBranchAddress("minMllAFOS", &minMllAFOS, &b_minMllAFOS);
   fChain->SetBranchAddress("maxMllAFOS", &maxMllAFOS, &b_maxMllAFOS);
   fChain->SetBranchAddress("minMllAFSS", &minMllAFSS, &b_minMllAFSS);
   fChain->SetBranchAddress("maxMllAFSS", &maxMllAFSS, &b_maxMllAFSS);
   fChain->SetBranchAddress("minMllAFAS", &minMllAFAS, &b_minMllAFAS);
   fChain->SetBranchAddress("maxMllAFAS", &maxMllAFAS, &b_maxMllAFAS);
   fChain->SetBranchAddress("m2l", &m2l, &b_m2l);
   fChain->SetBranchAddress("nBJet25", &nBJet25, &b_nBJet25);
   fChain->SetBranchAddress("nFatJet100", &nFatJet100, &b_nFatJet100);
   fChain->SetBranchAddress("nMuons10", &nMuons10, &b_nMuons10);
   fChain->SetBranchAddress("nElectrons10", &nElectrons10, &b_nElectrons10);
   fChain->SetBranchAddress("nTaus20", &nTaus20, &b_nTaus20);
   fChain->SetBranchAddress("nGammas20", &nGammas20, &b_nGammas20);
   fChain->SetBranchAddress("HLT_MonoJet", &HLT_MonoJet, &b_HLT_MonoJet);
   fChain->SetBranchAddress("Flag_EcalDeadCellTriggerPrimitiveFilter", &Flag_EcalDeadCellTriggerPrimitiveFilter, &b_Flag_EcalDeadCellTriggerPrimitiveFilter);
   fChain->SetBranchAddress("Flag_trkPOG_manystripclus53X", &Flag_trkPOG_manystripclus53X, &b_Flag_trkPOG_manystripclus53X);
   fChain->SetBranchAddress("Flag_ecalLaserCorrFilter", &Flag_ecalLaserCorrFilter, &b_Flag_ecalLaserCorrFilter);
   fChain->SetBranchAddress("Flag_trkPOG_toomanystripclus53X", &Flag_trkPOG_toomanystripclus53X, &b_Flag_trkPOG_toomanystripclus53X);
   fChain->SetBranchAddress("Flag_hcalLaserEventFilter", &Flag_hcalLaserEventFilter, &b_Flag_hcalLaserEventFilter);
   fChain->SetBranchAddress("Flag_trkPOG_logErrorTooManyClusters", &Flag_trkPOG_logErrorTooManyClusters, &b_Flag_trkPOG_logErrorTooManyClusters);
   fChain->SetBranchAddress("Flag_trkPOGFilters", &Flag_trkPOGFilters, &b_Flag_trkPOGFilters);
   fChain->SetBranchAddress("Flag_trackingFailureFilter", &Flag_trackingFailureFilter, &b_Flag_trackingFailureFilter);
   fChain->SetBranchAddress("Flag_CSCTightHaloFilter", &Flag_CSCTightHaloFilter, &b_Flag_CSCTightHaloFilter);
   fChain->SetBranchAddress("Flag_HBHENoiseFilter", &Flag_HBHENoiseFilter, &b_Flag_HBHENoiseFilter);
   fChain->SetBranchAddress("Flag_goodVertices", &Flag_goodVertices, &b_Flag_goodVertices);
   fChain->SetBranchAddress("Flag_METFilters", &Flag_METFilters, &b_Flag_METFilters);
   fChain->SetBranchAddress("Flag_eeBadScFilter", &Flag_eeBadScFilter, &b_Flag_eeBadScFilter);
   fChain->SetBranchAddress("met_pt", &met_pt, &b_met_pt);
   fChain->SetBranchAddress("met_eta", &met_eta, &b_met_eta);
   fChain->SetBranchAddress("met_phi", &met_phi, &b_met_phi);
   fChain->SetBranchAddress("met_mass", &met_mass, &b_met_mass);
   fChain->SetBranchAddress("met_sumEt", &met_sumEt, &b_met_sumEt);
   fChain->SetBranchAddress("met_genPt", &met_genPt, &b_met_genPt);
   fChain->SetBranchAddress("met_genPhi", &met_genPhi, &b_met_genPhi);
   fChain->SetBranchAddress("met_genEta", &met_genEta, &b_met_genEta);
   fChain->SetBranchAddress("metNoMu_pt", &metNoMu_pt, &b_metNoMu_pt);
   fChain->SetBranchAddress("metNoMu_eta", &metNoMu_eta, &b_metNoMu_eta);
   fChain->SetBranchAddress("metNoMu_phi", &metNoMu_phi, &b_metNoMu_phi);
   fChain->SetBranchAddress("metNoMu_mass", &metNoMu_mass, &b_metNoMu_mass);
   fChain->SetBranchAddress("metNoMu_sumEt", &metNoMu_sumEt, &b_metNoMu_sumEt);
   fChain->SetBranchAddress("metNoMu_genPt", &metNoMu_genPt, &b_metNoMu_genPt);
   fChain->SetBranchAddress("metNoMu_genPhi", &metNoMu_genPhi, &b_metNoMu_genPhi);
   fChain->SetBranchAddress("metNoMu_genEta", &metNoMu_genEta, &b_metNoMu_genEta);
   fChain->SetBranchAddress("metNoPU_pt", &metNoPU_pt, &b_metNoPU_pt);
   fChain->SetBranchAddress("metNoPU_eta", &metNoPU_eta, &b_metNoPU_eta);
   fChain->SetBranchAddress("metNoPU_phi", &metNoPU_phi, &b_metNoPU_phi);
   fChain->SetBranchAddress("metNoPU_mass", &metNoPU_mass, &b_metNoPU_mass);
   fChain->SetBranchAddress("nFatJet", &nFatJet, &b_nFatJet);
   fChain->SetBranchAddress("FatJet_id", FatJet_id, &b_FatJet_id);
   fChain->SetBranchAddress("FatJet_puId", FatJet_puId, &b_FatJet_puId);
   fChain->SetBranchAddress("FatJet_btagCSV", FatJet_btagCSV, &b_FatJet_btagCSV);
   fChain->SetBranchAddress("FatJet_btagCMVA", FatJet_btagCMVA, &b_FatJet_btagCMVA);
   fChain->SetBranchAddress("FatJet_rawPt", FatJet_rawPt, &b_FatJet_rawPt);
   fChain->SetBranchAddress("FatJet_mcPt", FatJet_mcPt, &b_FatJet_mcPt);
   fChain->SetBranchAddress("FatJet_mcFlavour", FatJet_mcFlavour, &b_FatJet_mcFlavour);
   fChain->SetBranchAddress("FatJet_mcMatchId", FatJet_mcMatchId, &b_FatJet_mcMatchId);
   fChain->SetBranchAddress("FatJet_pt", FatJet_pt, &b_FatJet_pt);
   fChain->SetBranchAddress("FatJet_eta", FatJet_eta, &b_FatJet_eta);
   fChain->SetBranchAddress("FatJet_phi", FatJet_phi, &b_FatJet_phi);
   fChain->SetBranchAddress("FatJet_mass", FatJet_mass, &b_FatJet_mass);
   fChain->SetBranchAddress("FatJet_prunedMass", FatJet_prunedMass, &b_FatJet_prunedMass);
   fChain->SetBranchAddress("FatJet_trimmedMass", FatJet_trimmedMass, &b_FatJet_trimmedMass);
   fChain->SetBranchAddress("FatJet_filteredMass", FatJet_filteredMass, &b_FatJet_filteredMass);
   fChain->SetBranchAddress("FatJet_tau1", FatJet_tau1, &b_FatJet_tau1);
   fChain->SetBranchAddress("FatJet_tau2", FatJet_tau2, &b_FatJet_tau2);
   fChain->SetBranchAddress("FatJet_tau3", FatJet_tau3, &b_FatJet_tau3);
   fChain->SetBranchAddress("FatJet_topMass", FatJet_topMass, &b_FatJet_topMass);
   fChain->SetBranchAddress("FatJet_minMass", FatJet_minMass, &b_FatJet_minMass);
   fChain->SetBranchAddress("FatJet_nSubJets", FatJet_nSubJets, &b_FatJet_nSubJets);
   fChain->SetBranchAddress("nGenPart", &nGenPart, &b_nGenPart);
   fChain->SetBranchAddress("GenPart_motherId", GenPart_motherId, &b_GenPart_motherId);
   fChain->SetBranchAddress("GenPart_grandmotherId", GenPart_grandmotherId, &b_GenPart_grandmotherId);
   fChain->SetBranchAddress("GenPart_sourceId", GenPart_sourceId, &b_GenPart_sourceId);
   fChain->SetBranchAddress("GenPart_charge", GenPart_charge, &b_GenPart_charge);
   fChain->SetBranchAddress("GenPart_status", GenPart_status, &b_GenPart_status);
   fChain->SetBranchAddress("GenPart_pdgId", GenPart_pdgId, &b_GenPart_pdgId);
   fChain->SetBranchAddress("GenPart_pt", GenPart_pt, &b_GenPart_pt);
   fChain->SetBranchAddress("GenPart_eta", GenPart_eta, &b_GenPart_eta);
   fChain->SetBranchAddress("GenPart_phi", GenPart_phi, &b_GenPart_phi);
   fChain->SetBranchAddress("GenPart_mass", GenPart_mass, &b_GenPart_mass);
   fChain->SetBranchAddress("GenPart_motherIndex", GenPart_motherIndex, &b_GenPart_motherIndex);
   fChain->SetBranchAddress("nisoTrack", &nisoTrack, &b_nisoTrack);
   fChain->SetBranchAddress("isoTrack_pdgId", isoTrack_pdgId, &b_isoTrack_pdgId);
   fChain->SetBranchAddress("isoTrack_pt", isoTrack_pt, &b_isoTrack_pt);
   fChain->SetBranchAddress("isoTrack_eta", isoTrack_eta, &b_isoTrack_eta);
   fChain->SetBranchAddress("isoTrack_phi", isoTrack_phi, &b_isoTrack_phi);
   fChain->SetBranchAddress("isoTrack_mass", isoTrack_mass, &b_isoTrack_mass);
   fChain->SetBranchAddress("isoTrack_charge", isoTrack_charge, &b_isoTrack_charge);
   fChain->SetBranchAddress("isoTrack_dz", isoTrack_dz, &b_isoTrack_dz);
   fChain->SetBranchAddress("isoTrack_absIso", isoTrack_absIso, &b_isoTrack_absIso);
   fChain->SetBranchAddress("isoTrack_mcMatchId", isoTrack_mcMatchId, &b_isoTrack_mcMatchId);
   fChain->SetBranchAddress("nLepGood", &nLepGood, &b_nLepGood);
   fChain->SetBranchAddress("LepGood_eleMVAId", LepGood_eleMVAId, &b_LepGood_eleMVAId);
   fChain->SetBranchAddress("LepGood_mvaId", LepGood_mvaId, &b_LepGood_mvaId);
   fChain->SetBranchAddress("LepGood_mvaIdTrig", LepGood_mvaIdTrig, &b_LepGood_mvaIdTrig);
   fChain->SetBranchAddress("LepGood_mvaIdPhys14", LepGood_mvaIdPhys14, &b_LepGood_mvaIdPhys14);
   fChain->SetBranchAddress("LepGood_mvaTTH", LepGood_mvaTTH, &b_LepGood_mvaTTH);
   fChain->SetBranchAddress("LepGood_mvaSusy", LepGood_mvaSusy, &b_LepGood_mvaSusy);
   fChain->SetBranchAddress("LepGood_jetPtRatio", LepGood_jetPtRatio, &b_LepGood_jetPtRatio);
   fChain->SetBranchAddress("LepGood_jetPtRel", LepGood_jetPtRel, &b_LepGood_jetPtRel);
   fChain->SetBranchAddress("LepGood_jetBTagCSV", LepGood_jetBTagCSV, &b_LepGood_jetBTagCSV);
   fChain->SetBranchAddress("LepGood_jetBTagCMVA", LepGood_jetBTagCMVA, &b_LepGood_jetBTagCMVA);
   fChain->SetBranchAddress("LepGood_jetDR", LepGood_jetDR, &b_LepGood_jetDR);
   fChain->SetBranchAddress("LepGood_charge", LepGood_charge, &b_LepGood_charge);
   fChain->SetBranchAddress("LepGood_tightId", LepGood_tightId, &b_LepGood_tightId);
   fChain->SetBranchAddress("LepGood_eleCutIdCSA14_25ns_v1", LepGood_eleCutIdCSA14_25ns_v1, &b_LepGood_eleCutIdCSA14_25ns_v1);
   fChain->SetBranchAddress("LepGood_eleCutIdCSA14_50ns_v1", LepGood_eleCutIdCSA14_50ns_v1, &b_LepGood_eleCutIdCSA14_50ns_v1);
   fChain->SetBranchAddress("LepGood_dxy", LepGood_dxy, &b_LepGood_dxy);
   fChain->SetBranchAddress("LepGood_dz", LepGood_dz, &b_LepGood_dz);
   fChain->SetBranchAddress("LepGood_edxy", LepGood_edxy, &b_LepGood_edxy);
   fChain->SetBranchAddress("LepGood_edz", LepGood_edz, &b_LepGood_edz);
   fChain->SetBranchAddress("LepGood_ip3d", LepGood_ip3d, &b_LepGood_ip3d);
   fChain->SetBranchAddress("LepGood_sip3d", LepGood_sip3d, &b_LepGood_sip3d);
   fChain->SetBranchAddress("LepGood_convVeto", LepGood_convVeto, &b_LepGood_convVeto);
   fChain->SetBranchAddress("LepGood_lostHits", LepGood_lostHits, &b_LepGood_lostHits);
   fChain->SetBranchAddress("LepGood_relIso03", LepGood_relIso03, &b_LepGood_relIso03);
   fChain->SetBranchAddress("LepGood_relIso04", LepGood_relIso04, &b_LepGood_relIso04);
   fChain->SetBranchAddress("LepGood_miniRelIso", LepGood_miniRelIso, &b_LepGood_miniRelIso);
   fChain->SetBranchAddress("LepGood_tightCharge", LepGood_tightCharge, &b_LepGood_tightCharge);
   fChain->SetBranchAddress("LepGood_mcMatchId", LepGood_mcMatchId, &b_LepGood_mcMatchId);
   fChain->SetBranchAddress("LepGood_mcMatchAny", LepGood_mcMatchAny, &b_LepGood_mcMatchAny);
   fChain->SetBranchAddress("LepGood_mcMatchTau", LepGood_mcMatchTau, &b_LepGood_mcMatchTau);
   fChain->SetBranchAddress("LepGood_mcPt", LepGood_mcPt, &b_LepGood_mcPt);
   fChain->SetBranchAddress("LepGood_mediumMuonId", LepGood_mediumMuonId, &b_LepGood_mediumMuonId);
   fChain->SetBranchAddress("LepGood_pdgId", LepGood_pdgId, &b_LepGood_pdgId);
   fChain->SetBranchAddress("LepGood_pt", LepGood_pt, &b_LepGood_pt);
   fChain->SetBranchAddress("LepGood_eta", LepGood_eta, &b_LepGood_eta);
   fChain->SetBranchAddress("LepGood_phi", LepGood_phi, &b_LepGood_phi);
   fChain->SetBranchAddress("LepGood_mass", LepGood_mass, &b_LepGood_mass);
   fChain->SetBranchAddress("LepGood_chargedHadRelIso03", LepGood_chargedHadRelIso03, &b_LepGood_chargedHadRelIso03);
   fChain->SetBranchAddress("LepGood_chargedHadRelIso04", LepGood_chargedHadRelIso04, &b_LepGood_chargedHadRelIso04);
   fChain->SetBranchAddress("LepGood_softMuonId", LepGood_softMuonId, &b_LepGood_softMuonId);
   fChain->SetBranchAddress("LepGood_pfMuonId", LepGood_pfMuonId, &b_LepGood_pfMuonId);
   fChain->SetBranchAddress("LepGood_eleCutId2012_full5x5", LepGood_eleCutId2012_full5x5, &b_LepGood_eleCutId2012_full5x5);
   fChain->SetBranchAddress("LepGood_trackerLayers", LepGood_trackerLayers, &b_LepGood_trackerLayers);
   fChain->SetBranchAddress("LepGood_pixelLayers", LepGood_pixelLayers, &b_LepGood_pixelLayers);
   fChain->SetBranchAddress("LepGood_trackerHits", LepGood_trackerHits, &b_LepGood_trackerHits);
   fChain->SetBranchAddress("LepGood_lostOuterHits", LepGood_lostOuterHits, &b_LepGood_lostOuterHits);
   fChain->SetBranchAddress("LepGood_innerTrackValidHitFraction", LepGood_innerTrackValidHitFraction, &b_LepGood_innerTrackValidHitFraction);
   fChain->SetBranchAddress("LepGood_innerTrackChi2", LepGood_innerTrackChi2, &b_LepGood_innerTrackChi2);
   fChain->SetBranchAddress("LepGood_nStations", LepGood_nStations, &b_LepGood_nStations);
   fChain->SetBranchAddress("LepGood_caloCompatibility", LepGood_caloCompatibility, &b_LepGood_caloCompatibility);
   fChain->SetBranchAddress("LepGood_globalTrackChi2", LepGood_globalTrackChi2, &b_LepGood_globalTrackChi2);
   fChain->SetBranchAddress("LepGood_trkKink", LepGood_trkKink, &b_LepGood_trkKink);
   fChain->SetBranchAddress("LepGood_segmentCompatibility", LepGood_segmentCompatibility, &b_LepGood_segmentCompatibility);
   fChain->SetBranchAddress("LepGood_chi2LocalPosition", LepGood_chi2LocalPosition, &b_LepGood_chi2LocalPosition);
   fChain->SetBranchAddress("LepGood_chi2LocalMomentum", LepGood_chi2LocalMomentum, &b_LepGood_chi2LocalMomentum);
   fChain->SetBranchAddress("LepGood_glbTrackProbability", LepGood_glbTrackProbability, &b_LepGood_glbTrackProbability);
   fChain->SetBranchAddress("LepGood_sigmaIEtaIEta", LepGood_sigmaIEtaIEta, &b_LepGood_sigmaIEtaIEta);
   fChain->SetBranchAddress("LepGood_dEtaScTrkIn", LepGood_dEtaScTrkIn, &b_LepGood_dEtaScTrkIn);
   fChain->SetBranchAddress("LepGood_dPhiScTrkIn", LepGood_dPhiScTrkIn, &b_LepGood_dPhiScTrkIn);
   fChain->SetBranchAddress("LepGood_hadronicOverEm", LepGood_hadronicOverEm, &b_LepGood_hadronicOverEm);
   fChain->SetBranchAddress("LepGood_eInvMinusPInv", LepGood_eInvMinusPInv, &b_LepGood_eInvMinusPInv);
   fChain->SetBranchAddress("LepGood_etaSc", LepGood_etaSc, &b_LepGood_etaSc);
   fChain->SetBranchAddress("LepGood_miniRelIsoCharged", LepGood_miniRelIsoCharged, &b_LepGood_miniRelIsoCharged);
   fChain->SetBranchAddress("LepGood_miniRelIsoNeutral", LepGood_miniRelIsoNeutral, &b_LepGood_miniRelIsoNeutral);
   fChain->SetBranchAddress("LepGood_hasSV", LepGood_hasSV, &b_LepGood_hasSV);
   fChain->SetBranchAddress("LepGood_svRedPt", LepGood_svRedPt, &b_LepGood_svRedPt);
   fChain->SetBranchAddress("LepGood_svRedM", LepGood_svRedM, &b_LepGood_svRedM);
   fChain->SetBranchAddress("LepGood_svLepSip3d", LepGood_svLepSip3d, &b_LepGood_svLepSip3d);
   fChain->SetBranchAddress("LepGood_svSip3d", LepGood_svSip3d, &b_LepGood_svSip3d);
   fChain->SetBranchAddress("LepGood_svNTracks", LepGood_svNTracks, &b_LepGood_svNTracks);
   fChain->SetBranchAddress("LepGood_svChi2n", LepGood_svChi2n, &b_LepGood_svChi2n);
   fChain->SetBranchAddress("LepGood_svDxy", LepGood_svDxy, &b_LepGood_svDxy);
   fChain->SetBranchAddress("LepGood_svMass", LepGood_svMass, &b_LepGood_svMass);
   fChain->SetBranchAddress("LepGood_svPt", LepGood_svPt, &b_LepGood_svPt);
   fChain->SetBranchAddress("LepGood_svMCMatchFraction", LepGood_svMCMatchFraction, &b_LepGood_svMCMatchFraction);
   fChain->SetBranchAddress("LepGood_svMva", LepGood_svMva, &b_LepGood_svMva);
   fChain->SetBranchAddress("LepGood_jetNDau", LepGood_jetNDau, &b_LepGood_jetNDau);
   fChain->SetBranchAddress("LepGood_jetNDauCharged", LepGood_jetNDauCharged, &b_LepGood_jetNDauCharged);
   fChain->SetBranchAddress("LepGood_jetNDauPV", LepGood_jetNDauPV, &b_LepGood_jetNDauPV);
   fChain->SetBranchAddress("LepGood_jetNDauNotPV", LepGood_jetNDauNotPV, &b_LepGood_jetNDauNotPV);
   fChain->SetBranchAddress("LepGood_jetmaxSignedSip3D", LepGood_jetmaxSignedSip3D, &b_LepGood_jetmaxSignedSip3D);
   fChain->SetBranchAddress("LepGood_jetmaxSip3D", LepGood_jetmaxSip3D, &b_LepGood_jetmaxSip3D);
   fChain->SetBranchAddress("LepGood_jetmaxSignedSip2D", LepGood_jetmaxSignedSip2D, &b_LepGood_jetmaxSignedSip2D);
   fChain->SetBranchAddress("LepGood_jetmaxSip2D", LepGood_jetmaxSip2D, &b_LepGood_jetmaxSip2D);
   fChain->SetBranchAddress("LepGood_jetPtRelv0", LepGood_jetPtRelv0, &b_LepGood_jetPtRelv0);
   fChain->SetBranchAddress("LepGood_jetMass", LepGood_jetMass, &b_LepGood_jetMass);
   fChain->SetBranchAddress("LepGood_jetPrunedMass", LepGood_jetPrunedMass, &b_LepGood_jetPrunedMass);
   fChain->SetBranchAddress("LepGood_jetDecDR", LepGood_jetDecDR, &b_LepGood_jetDecDR);
   fChain->SetBranchAddress("LepGood_jetDecPtRel", LepGood_jetDecPtRel, &b_LepGood_jetDecPtRel);
   fChain->SetBranchAddress("LepGood_jetDecPtRatio", LepGood_jetDecPtRatio, &b_LepGood_jetDecPtRatio);
   fChain->SetBranchAddress("LepGood_jetDecPrunedMass", LepGood_jetDecPrunedMass, &b_LepGood_jetDecPrunedMass);
   fChain->SetBranchAddress("LepGood_jetDecPrunedPtRatio", LepGood_jetDecPrunedPtRatio, &b_LepGood_jetDecPrunedPtRatio);
   fChain->SetBranchAddress("LepGood_jetDec02DR", LepGood_jetDec02DR, &b_LepGood_jetDec02DR);
   fChain->SetBranchAddress("LepGood_jetDec02PtRel", LepGood_jetDec02PtRel, &b_LepGood_jetDec02PtRel);
   fChain->SetBranchAddress("LepGood_jetDec02PtRatio", LepGood_jetDec02PtRatio, &b_LepGood_jetDec02PtRatio);
   fChain->SetBranchAddress("LepGood_jetDec02PrunedPtRatio", LepGood_jetDec02PrunedPtRatio, &b_LepGood_jetDec02PrunedPtRatio);
   fChain->SetBranchAddress("LepGood_jetDec02PrunedMass", LepGood_jetDec02PrunedMass, &b_LepGood_jetDec02PrunedMass);
   fChain->SetBranchAddress("ngenJet", &ngenJet, &b_ngenJet);
   fChain->SetBranchAddress("genJet_pdgId", genJet_pdgId, &b_genJet_pdgId);
   fChain->SetBranchAddress("genJet_pt", genJet_pt, &b_genJet_pt);
   fChain->SetBranchAddress("genJet_eta", genJet_eta, &b_genJet_eta);
   fChain->SetBranchAddress("genJet_phi", genJet_phi, &b_genJet_phi);
   fChain->SetBranchAddress("genJet_mass", genJet_mass, &b_genJet_mass);
   fChain->SetBranchAddress("genJet_charge", genJet_charge, &b_genJet_charge);
   fChain->SetBranchAddress("genJet_status", genJet_status, &b_genJet_status);
   fChain->SetBranchAddress("nJetFwd", &nJetFwd, &b_nJetFwd);
   fChain->SetBranchAddress("JetFwd_mcMatchFlav", JetFwd_mcMatchFlav, &b_JetFwd_mcMatchFlav);
   fChain->SetBranchAddress("JetFwd_area", JetFwd_area, &b_JetFwd_area);
   fChain->SetBranchAddress("JetFwd_qgl", JetFwd_qgl, &b_JetFwd_qgl);
   fChain->SetBranchAddress("JetFwd_ptd", JetFwd_ptd, &b_JetFwd_ptd);
   fChain->SetBranchAddress("JetFwd_axis2", JetFwd_axis2, &b_JetFwd_axis2);
   fChain->SetBranchAddress("JetFwd_mult", JetFwd_mult, &b_JetFwd_mult);
   fChain->SetBranchAddress("JetFwd_partonId", JetFwd_partonId, &b_JetFwd_partonId);
   fChain->SetBranchAddress("JetFwd_partonMotherId", JetFwd_partonMotherId, &b_JetFwd_partonMotherId);
   fChain->SetBranchAddress("JetFwd_nLeptons", JetFwd_nLeptons, &b_JetFwd_nLeptons);
   fChain->SetBranchAddress("JetFwd_id", JetFwd_id, &b_JetFwd_id);
   fChain->SetBranchAddress("JetFwd_puId", JetFwd_puId, &b_JetFwd_puId);
   fChain->SetBranchAddress("JetFwd_btagCSV", JetFwd_btagCSV, &b_JetFwd_btagCSV);
   fChain->SetBranchAddress("JetFwd_btagCMVA", JetFwd_btagCMVA, &b_JetFwd_btagCMVA);
   fChain->SetBranchAddress("JetFwd_rawPt", JetFwd_rawPt, &b_JetFwd_rawPt);
   fChain->SetBranchAddress("JetFwd_mcPt", JetFwd_mcPt, &b_JetFwd_mcPt);
   fChain->SetBranchAddress("JetFwd_mcFlavour", JetFwd_mcFlavour, &b_JetFwd_mcFlavour);
   fChain->SetBranchAddress("JetFwd_mcMatchId", JetFwd_mcMatchId, &b_JetFwd_mcMatchId);
   fChain->SetBranchAddress("JetFwd_pt", JetFwd_pt, &b_JetFwd_pt);
   fChain->SetBranchAddress("JetFwd_eta", JetFwd_eta, &b_JetFwd_eta);
   fChain->SetBranchAddress("JetFwd_phi", JetFwd_phi, &b_JetFwd_phi);
   fChain->SetBranchAddress("JetFwd_mass", JetFwd_mass, &b_JetFwd_mass);
   fChain->SetBranchAddress("JetFwd_prunedMass", JetFwd_prunedMass, &b_JetFwd_prunedMass);
   fChain->SetBranchAddress("JetFwd_mcNumPartons", JetFwd_mcNumPartons, &b_JetFwd_mcNumPartons);
   fChain->SetBranchAddress("JetFwd_mcNumLeptons", JetFwd_mcNumLeptons, &b_JetFwd_mcNumLeptons);
   fChain->SetBranchAddress("JetFwd_mcNumTaus", JetFwd_mcNumTaus, &b_JetFwd_mcNumTaus);
   fChain->SetBranchAddress("JetFwd_mcAnyPartonMass", JetFwd_mcAnyPartonMass, &b_JetFwd_mcAnyPartonMass);
   fChain->SetBranchAddress("JetFwd_nSubJets", JetFwd_nSubJets, &b_JetFwd_nSubJets);
   fChain->SetBranchAddress("JetFwd_nSubJets25", JetFwd_nSubJets25, &b_JetFwd_nSubJets25);
   fChain->SetBranchAddress("JetFwd_nSubJets30", JetFwd_nSubJets30, &b_JetFwd_nSubJets30);
   fChain->SetBranchAddress("JetFwd_nSubJets40", JetFwd_nSubJets40, &b_JetFwd_nSubJets40);
   fChain->SetBranchAddress("JetFwd_nSubJetsZ01", JetFwd_nSubJetsZ01, &b_JetFwd_nSubJetsZ01);
   fChain->SetBranchAddress("JetFwd_chHEF", JetFwd_chHEF, &b_JetFwd_chHEF);
   fChain->SetBranchAddress("JetFwd_neHEF", JetFwd_neHEF, &b_JetFwd_neHEF);
   fChain->SetBranchAddress("JetFwd_phEF", JetFwd_phEF, &b_JetFwd_phEF);
   fChain->SetBranchAddress("JetFwd_eEF", JetFwd_eEF, &b_JetFwd_eEF);
   fChain->SetBranchAddress("JetFwd_muEF", JetFwd_muEF, &b_JetFwd_muEF);
   fChain->SetBranchAddress("JetFwd_HFHEF", JetFwd_HFHEF, &b_JetFwd_HFHEF);
   fChain->SetBranchAddress("JetFwd_HFEMEF", JetFwd_HFEMEF, &b_JetFwd_HFEMEF);
   fChain->SetBranchAddress("JetFwd_chHMult", JetFwd_chHMult, &b_JetFwd_chHMult);
   fChain->SetBranchAddress("JetFwd_neHMult", JetFwd_neHMult, &b_JetFwd_neHMult);
   fChain->SetBranchAddress("JetFwd_phMult", JetFwd_phMult, &b_JetFwd_phMult);
   fChain->SetBranchAddress("JetFwd_eMult", JetFwd_eMult, &b_JetFwd_eMult);
   fChain->SetBranchAddress("JetFwd_muMult", JetFwd_muMult, &b_JetFwd_muMult);
   fChain->SetBranchAddress("JetFwd_HFHMult", JetFwd_HFHMult, &b_JetFwd_HFHMult);
   fChain->SetBranchAddress("JetFwd_HFEMMult", JetFwd_HFEMMult, &b_JetFwd_HFEMMult);
   fChain->SetBranchAddress("nGammaGood", &nGammaGood, &b_nGammaGood);
   fChain->SetBranchAddress("GammaGood_idCutBased", &GammaGood_idCutBased, &b_GammaGood_idCutBased);
   fChain->SetBranchAddress("GammaGood_hOverE", &GammaGood_hOverE, &b_GammaGood_hOverE);
   fChain->SetBranchAddress("GammaGood_r9", &GammaGood_r9, &b_GammaGood_r9);
   fChain->SetBranchAddress("GammaGood_sigmaIetaIeta", &GammaGood_sigmaIetaIeta, &b_GammaGood_sigmaIetaIeta);
   fChain->SetBranchAddress("GammaGood_chHadIso", &GammaGood_chHadIso, &b_GammaGood_chHadIso);
   fChain->SetBranchAddress("GammaGood_chHadIso04", &GammaGood_chHadIso04, &b_GammaGood_chHadIso04);
   fChain->SetBranchAddress("GammaGood_neuHadIso", &GammaGood_neuHadIso, &b_GammaGood_neuHadIso);
   fChain->SetBranchAddress("GammaGood_phIso", &GammaGood_phIso, &b_GammaGood_phIso);
   fChain->SetBranchAddress("GammaGood_mcMatchId", &GammaGood_mcMatchId, &b_GammaGood_mcMatchId);
   fChain->SetBranchAddress("GammaGood_mcPt", &GammaGood_mcPt, &b_GammaGood_mcPt);
   fChain->SetBranchAddress("GammaGood_pdgId", &GammaGood_pdgId, &b_GammaGood_pdgId);
   fChain->SetBranchAddress("GammaGood_pt", &GammaGood_pt, &b_GammaGood_pt);
   fChain->SetBranchAddress("GammaGood_eta", &GammaGood_eta, &b_GammaGood_eta);
   fChain->SetBranchAddress("GammaGood_phi", &GammaGood_phi, &b_GammaGood_phi);
   fChain->SetBranchAddress("GammaGood_mass", &GammaGood_mass, &b_GammaGood_mass);
   fChain->SetBranchAddress("GammaGood_genIso04", &GammaGood_genIso04, &b_GammaGood_genIso04);
   fChain->SetBranchAddress("GammaGood_genIso", &GammaGood_genIso, &b_GammaGood_genIso);
   fChain->SetBranchAddress("GammaGood_chHadIsoRC04", &GammaGood_chHadIsoRC04, &b_GammaGood_chHadIsoRC04);
   fChain->SetBranchAddress("GammaGood_chHadIsoRC", &GammaGood_chHadIsoRC, &b_GammaGood_chHadIsoRC);
   fChain->SetBranchAddress("nTauGood", &nTauGood, &b_nTauGood);
   fChain->SetBranchAddress("TauGood_charge", TauGood_charge, &b_TauGood_charge);
   fChain->SetBranchAddress("TauGood_decayMode", TauGood_decayMode, &b_TauGood_decayMode);
   fChain->SetBranchAddress("TauGood_idDecayMode", TauGood_idDecayMode, &b_TauGood_idDecayMode);
   fChain->SetBranchAddress("TauGood_idDecayModeNewDMs", TauGood_idDecayModeNewDMs, &b_TauGood_idDecayModeNewDMs);
   fChain->SetBranchAddress("TauGood_dxy", TauGood_dxy, &b_TauGood_dxy);
   fChain->SetBranchAddress("TauGood_dz", TauGood_dz, &b_TauGood_dz);
   fChain->SetBranchAddress("TauGood_idMVA", TauGood_idMVA, &b_TauGood_idMVA);
   fChain->SetBranchAddress("TauGood_idMVANewDM", TauGood_idMVANewDM, &b_TauGood_idMVANewDM);
   fChain->SetBranchAddress("TauGood_idCI3hit", TauGood_idCI3hit, &b_TauGood_idCI3hit);
   fChain->SetBranchAddress("TauGood_idAntiMu", TauGood_idAntiMu, &b_TauGood_idAntiMu);
   fChain->SetBranchAddress("TauGood_idAntiE", TauGood_idAntiE, &b_TauGood_idAntiE);
   fChain->SetBranchAddress("TauGood_isoCI3hit", TauGood_isoCI3hit, &b_TauGood_isoCI3hit);
   fChain->SetBranchAddress("TauGood_mcMatchId", TauGood_mcMatchId, &b_TauGood_mcMatchId);
   fChain->SetBranchAddress("TauGood_pdgId", TauGood_pdgId, &b_TauGood_pdgId);
   fChain->SetBranchAddress("TauGood_pt", TauGood_pt, &b_TauGood_pt);
   fChain->SetBranchAddress("TauGood_eta", TauGood_eta, &b_TauGood_eta);
   fChain->SetBranchAddress("TauGood_phi", TauGood_phi, &b_TauGood_phi);
   fChain->SetBranchAddress("TauGood_mass", TauGood_mass, &b_TauGood_mass);
   fChain->SetBranchAddress("ngenLep", &ngenLep, &b_ngenLep);
   fChain->SetBranchAddress("genLep_motherId", &genLep_motherId, &b_genLep_motherId);
   fChain->SetBranchAddress("genLep_grandmotherId", &genLep_grandmotherId, &b_genLep_grandmotherId);
   fChain->SetBranchAddress("genLep_sourceId", &genLep_sourceId, &b_genLep_sourceId);
   fChain->SetBranchAddress("genLep_charge", &genLep_charge, &b_genLep_charge);
   fChain->SetBranchAddress("genLep_status", &genLep_status, &b_genLep_status);
   fChain->SetBranchAddress("genLep_pdgId", &genLep_pdgId, &b_genLep_pdgId);
   fChain->SetBranchAddress("genLep_pt", &genLep_pt, &b_genLep_pt);
   fChain->SetBranchAddress("genLep_eta", &genLep_eta, &b_genLep_eta);
   fChain->SetBranchAddress("genLep_phi", &genLep_phi, &b_genLep_phi);
   fChain->SetBranchAddress("genLep_mass", &genLep_mass, &b_genLep_mass);
   fChain->SetBranchAddress("genLep_motherIndex", &genLep_motherIndex, &b_genLep_motherIndex);
   fChain->SetBranchAddress("nLHEweight", &nLHEweight, &b_nLHEweight);
   fChain->SetBranchAddress("LHEweight_id", &LHEweight_id, &b_LHEweight_id);
   fChain->SetBranchAddress("LHEweight_wgt", &LHEweight_wgt, &b_LHEweight_wgt);
   fChain->SetBranchAddress("nJet", &nJet, &b_nJet);
   fChain->SetBranchAddress("Jet_mcMatchFlav", Jet_mcMatchFlav, &b_Jet_mcMatchFlav);
   fChain->SetBranchAddress("Jet_area", Jet_area, &b_Jet_area);
   fChain->SetBranchAddress("Jet_qgl", Jet_qgl, &b_Jet_qgl);
   fChain->SetBranchAddress("Jet_ptd", Jet_ptd, &b_Jet_ptd);
   fChain->SetBranchAddress("Jet_axis2", Jet_axis2, &b_Jet_axis2);
   fChain->SetBranchAddress("Jet_mult", Jet_mult, &b_Jet_mult);
   fChain->SetBranchAddress("Jet_partonId", Jet_partonId, &b_Jet_partonId);
   fChain->SetBranchAddress("Jet_partonMotherId", Jet_partonMotherId, &b_Jet_partonMotherId);
   fChain->SetBranchAddress("Jet_nLeptons", Jet_nLeptons, &b_Jet_nLeptons);
   fChain->SetBranchAddress("Jet_id", Jet_id, &b_Jet_id);
   fChain->SetBranchAddress("Jet_puId", Jet_puId, &b_Jet_puId);
   fChain->SetBranchAddress("Jet_btagCSV", Jet_btagCSV, &b_Jet_btagCSV);
   fChain->SetBranchAddress("Jet_btagCMVA", Jet_btagCMVA, &b_Jet_btagCMVA);
   fChain->SetBranchAddress("Jet_rawPt", Jet_rawPt, &b_Jet_rawPt);
   fChain->SetBranchAddress("Jet_mcPt", Jet_mcPt, &b_Jet_mcPt);
   fChain->SetBranchAddress("Jet_mcFlavour", Jet_mcFlavour, &b_Jet_mcFlavour);
   fChain->SetBranchAddress("Jet_mcMatchId", Jet_mcMatchId, &b_Jet_mcMatchId);
   fChain->SetBranchAddress("Jet_pt", Jet_pt, &b_Jet_pt);
   fChain->SetBranchAddress("Jet_eta", Jet_eta, &b_Jet_eta);
   fChain->SetBranchAddress("Jet_phi", Jet_phi, &b_Jet_phi);
   fChain->SetBranchAddress("Jet_mass", Jet_mass, &b_Jet_mass);
   fChain->SetBranchAddress("Jet_prunedMass", Jet_prunedMass, &b_Jet_prunedMass);
   fChain->SetBranchAddress("Jet_mcNumPartons", Jet_mcNumPartons, &b_Jet_mcNumPartons);
   fChain->SetBranchAddress("Jet_mcNumLeptons", Jet_mcNumLeptons, &b_Jet_mcNumLeptons);
   fChain->SetBranchAddress("Jet_mcNumTaus", Jet_mcNumTaus, &b_Jet_mcNumTaus);
   fChain->SetBranchAddress("Jet_mcAnyPartonMass", Jet_mcAnyPartonMass, &b_Jet_mcAnyPartonMass);
   fChain->SetBranchAddress("Jet_nSubJets", Jet_nSubJets, &b_Jet_nSubJets);
   fChain->SetBranchAddress("Jet_nSubJets25", Jet_nSubJets25, &b_Jet_nSubJets25);
   fChain->SetBranchAddress("Jet_nSubJets30", Jet_nSubJets30, &b_Jet_nSubJets30);
   fChain->SetBranchAddress("Jet_nSubJets40", Jet_nSubJets40, &b_Jet_nSubJets40);
   fChain->SetBranchAddress("Jet_nSubJetsZ01", Jet_nSubJetsZ01, &b_Jet_nSubJetsZ01);
   fChain->SetBranchAddress("Jet_chHEF", Jet_chHEF, &b_Jet_chHEF);
   fChain->SetBranchAddress("Jet_neHEF", Jet_neHEF, &b_Jet_neHEF);
   fChain->SetBranchAddress("Jet_phEF", Jet_phEF, &b_Jet_phEF);
   fChain->SetBranchAddress("Jet_eEF", Jet_eEF, &b_Jet_eEF);
   fChain->SetBranchAddress("Jet_muEF", Jet_muEF, &b_Jet_muEF);
   fChain->SetBranchAddress("Jet_HFHEF", Jet_HFHEF, &b_Jet_HFHEF);
   fChain->SetBranchAddress("Jet_HFEMEF", Jet_HFEMEF, &b_Jet_HFEMEF);
   fChain->SetBranchAddress("Jet_chHMult", Jet_chHMult, &b_Jet_chHMult);
   fChain->SetBranchAddress("Jet_neHMult", Jet_neHMult, &b_Jet_neHMult);
   fChain->SetBranchAddress("Jet_phMult", Jet_phMult, &b_Jet_phMult);
   fChain->SetBranchAddress("Jet_eMult", Jet_eMult, &b_Jet_eMult);
   fChain->SetBranchAddress("Jet_muMult", Jet_muMult, &b_Jet_muMult);
   fChain->SetBranchAddress("Jet_HFHMult", Jet_HFHMult, &b_Jet_HFHMult);
   fChain->SetBranchAddress("Jet_HFEMMult", Jet_HFEMMult, &b_Jet_HFEMMult);
   Notify();
}

Bool_t testfriend::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void testfriend::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t testfriend::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef testfriend_cxx
