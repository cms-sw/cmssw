#ifndef TreeAnalysisReadGen_h
#define TreeAnalysisReadGen_h
//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sat Nov 14 12:12:00 2009 by ROOT version 5.23/02
// from TTree tree/tree
//////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TDirectory.h"
#include "TH1F.h"
#include "TString.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <cmath>

class TreeAnalysisReadGen {

public :

  enum l1decision {L1SingleJet,L1SingleTauJet, L1SingleIsoEG,L1SingleEG,L1SingleMu};

  std::map<std::string, int> l1Names;
  std::string                l1Name;
  static const int NEtaBins = 4;
  static const int NPBins   = 21;
  double genPartPBins[22], genPartEtaBins[5];
  TFile *fout;
  double dRCut;
  // inclusive distributions
  static const int PTypes = 3;
  TH1F *h_trkPAll[PTypes],      *h_trkPtAll[PTypes],      *h_trkEtaAll[PTypes],      *h_trkPhiAll[PTypes];
  TH1F *h_trkPIsoNxN[PTypes], *h_trkPtIsoNxN[PTypes], *h_trkEtaIsoNxN[PTypes], *h_trkPhiIsoNxN[PTypes];
  TH1F *h_trkPIsoR[PTypes], *h_trkPtIsoR[PTypes], *h_trkEtaIsoR[PTypes], *h_trkPhiIsoR[PTypes];
  TH1F *h_trkDEta[NPBins][NEtaBins], *h_trkDPhi[NPBins][NEtaBins];
  
  TH1F *h_L1CenJetPt, *h_L1FwdJetPt, *h_L1TauJetPt;
  TH1F *h_L1LeadJetPt;

  TH1F *h_L1Decision;
  TH1F *h_L1_iso31x31[NPBins][NEtaBins];  
  TH1F *h_L1_iso31x31_isoPhoton_11x11_1[NPBins][NEtaBins];  
  TH1F *h_L1_iso31x31_isoPhoton_11x11_2[NPBins][NEtaBins];  
  TH1F *h_L1_iso31x31_isoNeutral_11x11_1[NPBins][NEtaBins];  
  TH1F *h_L1_iso31x31_isoNeutral_11x11_2[NPBins][NEtaBins];  

  TH1F *h_maxNearP31x31[NPBins][NEtaBins],
       *h_maxNearP25x25[NPBins][NEtaBins], 
       *h_maxNearP21x21[NPBins][NEtaBins], 
       *h_maxNearP15x15[NPBins][NEtaBins], 
       *h_maxNearP11x11[NPBins][NEtaBins];
  TH1F *h_trkP_iso31x31,
       *h_trkP_iso25x25,
       *h_trkP_iso21x21,
       *h_trkP_iso15x15,
       *h_trkP_iso11x11;
  TH1F *h_photon_iso31x31[NPBins][NEtaBins],
       *h_photon_iso25x25[NPBins][NEtaBins], 
       *h_photon_iso21x21[NPBins][NEtaBins], 
       *h_photon_iso15x15[NPBins][NEtaBins], 
       *h_photon_iso11x11[NPBins][NEtaBins];
  TH1F *h_charged_iso31x31[NPBins][NEtaBins],
       *h_charged_iso25x25[NPBins][NEtaBins], 
       *h_charged_iso21x21[NPBins][NEtaBins], 
       *h_charged_iso15x15[NPBins][NEtaBins], 
       *h_charged_iso11x11[NPBins][NEtaBins];
  TH1F *h_neutral_iso31x31[NPBins][NEtaBins],
       *h_neutral_iso25x25[NPBins][NEtaBins], 
       *h_neutral_iso21x21[NPBins][NEtaBins], 
       *h_neutral_iso15x15[NPBins][NEtaBins], 
       *h_neutral_iso11x11[NPBins][NEtaBins];
  TH1F *h_contamination_iso31x31[NPBins][NEtaBins],
       *h_contamination_iso25x25[NPBins][NEtaBins], 
       *h_contamination_iso21x21[NPBins][NEtaBins], 
       *h_contamination_iso15x15[NPBins][NEtaBins], 
       *h_contamination_iso11x11[NPBins][NEtaBins];
  TH1F *h_photon11x11_iso31x31[NPBins][NEtaBins],
       *h_charged11x11_iso31x31[NPBins][NEtaBins],
       *h_neutral11x11_iso31x31[NPBins][NEtaBins],
       *h_contamination11x11_iso31x31[NPBins][NEtaBins];
  TH1F *h_photon11x11_isoEcal_NxN[NPBins][NEtaBins],
       *h_charged11x11_isoEcal_NxN[NPBins][NEtaBins],
       *h_neutral11x11_isoEcal_NxN[NPBins][NEtaBins],
       *h_contamination11x11_isoEcal_NxN[NPBins][NEtaBins];
  TH1F *h_photonR_isoEcal_R[NPBins][NEtaBins],
       *h_chargedR_isoEcal_R[NPBins][NEtaBins],
       *h_neutralR_isoEcal_R[NPBins][NEtaBins],
       *h_contaminationR_isoEcal_R[NPBins][NEtaBins];
  TH1F *h_photonHC5x5_IsoNxN[NPBins][NEtaBins],
       *h_chargedHC5x5_IsoNxN[NPBins][NEtaBins],
       *h_neutralHC5x5_IsoNxN[NPBins][NEtaBins],
       *h_contaminationHC5x5_IsoNxN[NPBins][NEtaBins];
  TH1F *h_photonHCR_IsoR[NPBins][NEtaBins],
       *h_chargedHCR_IsoR[NPBins][NEtaBins],
       *h_neutralHCR_IsoR[NPBins][NEtaBins],
       *h_contaminationHCR_IsoR[NPBins][NEtaBins];


  TH1F *h_pdgId_iso31x31[NPBins][NEtaBins];
  

  TChain          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  std::vector<double>  *t_isoTrkPAll;
  std::vector<double>  *t_isoTrkPtAll;
  std::vector<double>  *t_isoTrkPhiAll;
  std::vector<double>  *t_isoTrkEtaAll;
  std::vector<double>  *t_isoTrkDPhiAll;
  std::vector<double>  *t_isoTrkDEtaAll;
  std::vector<double>  *t_isoTrkPdgIdAll;
  std::vector<double>  *t_isoTrkP;
  std::vector<double>  *t_isoTrkPt;
  std::vector<double>  *t_isoTrkEne;
  std::vector<double>  *t_isoTrkEta;
  std::vector<double>  *t_isoTrkPhi;
  std::vector<double>  *t_isoTrkPdgId;
  std::vector<double>  *t_maxNearP31x31;
  std::vector<double>  *t_cHadronEne31x31;
  std::vector<double>  *t_cHadronEne31x31_1;
  std::vector<double>  *t_cHadronEne31x31_2;
  std::vector<double>  *t_cHadronEne31x31_3;
  std::vector<double>  *t_nHadronEne31x31;
  std::vector<double>  *t_photonEne31x31;
  std::vector<double>  *t_eleEne31x31;
  std::vector<double>  *t_muEne31x31;
  std::vector<double>  *t_maxNearP25x25;
  std::vector<double>  *t_cHadronEne25x25;
  std::vector<double>  *t_cHadronEne25x25_1;
  std::vector<double>  *t_cHadronEne25x25_2;
  std::vector<double>  *t_cHadronEne25x25_3;
  std::vector<double>  *t_nHadronEne25x25;
  std::vector<double>  *t_photonEne25x25;
  std::vector<double>  *t_eleEne25x25;
  std::vector<double>  *t_muEne25x25;
  std::vector<double>  *t_maxNearP21x21;
  std::vector<double>  *t_cHadronEne21x21;
  std::vector<double>  *t_cHadronEne21x21_1;
  std::vector<double>  *t_cHadronEne21x21_2;
  std::vector<double>  *t_cHadronEne21x21_3;
  std::vector<double>  *t_nHadronEne21x21;
  std::vector<double>  *t_photonEne21x21;
  std::vector<double>  *t_eleEne21x21;
  std::vector<double>  *t_muEne21x21;
  std::vector<double>  *t_maxNearP15x15;
  std::vector<double>  *t_cHadronEne15x15;
  std::vector<double>  *t_cHadronEne15x15_1;
  std::vector<double>  *t_cHadronEne15x15_2;
  std::vector<double>  *t_cHadronEne15x15_3;
  std::vector<double>  *t_nHadronEne15x15;
  std::vector<double>  *t_photonEne15x15;
  std::vector<double>  *t_eleEne15x15;
  std::vector<double>  *t_muEne15x15;
  std::vector<double>  *t_maxNearP11x11;
  std::vector<double>  *t_cHadronEne11x11;
  std::vector<double>  *t_cHadronEne11x11_1;
  std::vector<double>  *t_cHadronEne11x11_2;
  std::vector<double>  *t_cHadronEne11x11_3;
  std::vector<double>  *t_nHadronEne11x11;
  std::vector<double>  *t_photonEne11x11;
  std::vector<double>  *t_eleEne11x11;
  std::vector<double>  *t_muEne11x11;
  std::vector<double>  *t_maxNearP9x9;
  std::vector<double>  *t_cHadronEne9x9;
  std::vector<double>  *t_cHadronEne9x9_1;
  std::vector<double>  *t_cHadronEne9x9_2;
  std::vector<double>  *t_cHadronEne9x9_3;
  std::vector<double>  *t_nHadronEne9x9;
  std::vector<double>  *t_photonEne9x9;
  std::vector<double>  *t_eleEne9x9;
  std::vector<double>  *t_muEne9x9;
  std::vector<double>  *t_maxNearP7x7;
  std::vector<double>  *t_cHadronEne7x7;
  std::vector<double>  *t_cHadronEne7x7_1;
  std::vector<double>  *t_cHadronEne7x7_2;
  std::vector<double>  *t_cHadronEne7x7_3;
  std::vector<double>  *t_nHadronEne7x7;
  std::vector<double>  *t_photonEne7x7;
  std::vector<double>  *t_eleEne7x7;
  std::vector<double>  *t_muEne7x7;
  std::vector<double>  *t_maxNearPHC3x3;
  std::vector<double>  *t_cHadronEneHC3x3;
  std::vector<double>  *t_cHadronEneHC3x3_1;
  std::vector<double>  *t_cHadronEneHC3x3_2;
  std::vector<double>  *t_cHadronEneHC3x3_3;
  std::vector<double>  *t_nHadronEneHC3x3;
  std::vector<double>  *t_photonEneHC3x3;
  std::vector<double>  *t_eleEneHC3x3;
  std::vector<double>  *t_muEneHC3x3;
  std::vector<double>  *t_maxNearPHC5x5;
  std::vector<double>  *t_cHadronEneHC5x5;
  std::vector<double>  *t_cHadronEneHC5x5_1;
  std::vector<double>  *t_cHadronEneHC5x5_2;
  std::vector<double>  *t_cHadronEneHC5x5_3;
  std::vector<double>  *t_nHadronEneHC5x5;
  std::vector<double>  *t_photonEneHC5x5;
  std::vector<double>  *t_eleEneHC5x5;
  std::vector<double>  *t_muEneHC5x5;
  std::vector<double>  *t_maxNearPHC7x7;
  std::vector<double>  *t_cHadronEneHC7x7;
  std::vector<double>  *t_cHadronEneHC7x7_1;
  std::vector<double>  *t_cHadronEneHC7x7_2;
  std::vector<double>  *t_cHadronEneHC7x7_3;
  std::vector<double>  *t_nHadronEneHC7x7;
  std::vector<double>  *t_photonEneHC7x7;
  std::vector<double>  *t_eleEneHC7x7;
  std::vector<double>  *t_muEneHC7x7;
  std::vector<double>  *t_maxNearPR;
  std::vector<double>  *t_cHadronEneR;
  std::vector<double>  *t_cHadronEneR_1;
  std::vector<double>  *t_cHadronEneR_2;
  std::vector<double>  *t_cHadronEneR_3;
  std::vector<double>  *t_nHadronEneR;
  std::vector<double>  *t_photonEneR;
  std::vector<double>  *t_eleEneR;
  std::vector<double>  *t_muEneR;
  std::vector<double>  *t_maxNearPIsoR;
  std::vector<double>  *t_cHadronEneIsoR;
  std::vector<double>  *t_cHadronEneIsoR_1;
  std::vector<double>  *t_cHadronEneIsoR_2;
  std::vector<double>  *t_cHadronEneIsoR_3;
  std::vector<double>  *t_nHadronEneIsoR;
  std::vector<double>  *t_photonEneIsoR;
  std::vector<double>  *t_eleEneIsoR;
  std::vector<double>  *t_muEneIsoR;
  std::vector<double>  *t_maxNearPHCR;
  std::vector<double>  *t_cHadronEneHCR;
  std::vector<double>  *t_cHadronEneHCR_1;
  std::vector<double>  *t_cHadronEneHCR_2;
  std::vector<double>  *t_cHadronEneHCR_3;
  std::vector<double>  *t_nHadronEneHCR;
  std::vector<double>  *t_photonEneHCR;
  std::vector<double>  *t_eleEneHCR;
  std::vector<double>  *t_muEneHCR;
  std::vector<double>  *t_maxNearPIsoHCR;
  std::vector<double>  *t_cHadronEneIsoHCR;
  std::vector<double>  *t_cHadronEneIsoHCR_1;
  std::vector<double>  *t_cHadronEneIsoHCR_2;
  std::vector<double>  *t_cHadronEneIsoHCR_3;
  std::vector<double>  *t_nHadronEneIsoHCR;
  std::vector<double>  *t_photonEneIsoHCR;
  std::vector<double>  *t_eleEneIsoHCR;
  std::vector<double>  *t_muEneIsoHCR;
  std::vector<int>     *t_L1Decision;
  std::vector<double>  *t_L1CenJetPt;
  std::vector<double>  *t_L1CenJetEta;
  std::vector<double>  *t_L1CenJetPhi;
  std::vector<double>  *t_L1FwdJetPt;
  std::vector<double>  *t_L1FwdJetEta;
  std::vector<double>  *t_L1FwdJetPhi;
  std::vector<double>  *t_L1TauJetPt;
  std::vector<double>  *t_L1TauJetEta;
  std::vector<double>  *t_L1TauJetPhi;
  std::vector<double>  *t_L1MuonPt;
  std::vector<double>  *t_L1MuonEta;
  std::vector<double>  *t_L1MuonPhi;
  std::vector<double>  *t_L1IsoEMPt;
  std::vector<double>  *t_L1IsoEMEta;
  std::vector<double>  *t_L1IsoEMPhi;
  std::vector<double>  *t_L1NonIsoEMPt;
  std::vector<double>  *t_L1NonIsoEMEta;
  std::vector<double>  *t_L1NonIsoEMPhi;
  std::vector<double>  *t_L1METPt;
  std::vector<double>  *t_L1METEta;
  std::vector<double>  *t_L1METPhi;
  
  // List of branches
  TBranch        *b_t_isoTrkPAll;   //!
  TBranch        *b_t_isoTrkPtAll;   //!
  TBranch        *b_t_isoTrkPhiAll;   //!
  TBranch        *b_t_isoTrkEtaAll;   //!
  TBranch        *b_t_isoTrkDPhiAll;   //!
  TBranch        *b_t_isoTrkDEtaAll;   //!
  TBranch        *b_t_isoTrkPdgIdAll;   //!
  TBranch        *b_t_isoTrkP;   //!
  TBranch        *b_t_isoTrkPt;   //!
  TBranch        *b_t_isoTrkEne;   //!
  TBranch        *b_t_isoTrkEta;   //!
  TBranch        *b_t_isoTrkPhi;   //!
  TBranch        *b_t_isoTrkPdgId;   //!
  TBranch        *b_t_maxNearP31x31;   //!
  TBranch        *b_t_cHadronEne31x31;   //!
  TBranch        *b_t_cHadronEne31x31_1;   //!
  TBranch        *b_t_cHadronEne31x31_2;   //!
  TBranch        *b_t_cHadronEne31x31_3;   //!
  TBranch        *b_t_nHadronEne31x31;   //!
  TBranch        *b_t_photonEne31x31;   //!
  TBranch        *b_t_eleEne31x31;   //!
  TBranch        *b_t_muEne31x31;   //!
  TBranch        *b_t_maxNearP25x25;   //!
  TBranch        *b_t_cHadronEne25x25;   //!
  TBranch        *b_t_cHadronEne25x25_1;   //!
  TBranch        *b_t_cHadronEne25x25_2;   //!
  TBranch        *b_t_cHadronEne25x25_3;   //!
  TBranch        *b_t_nHadronEne25x25;   //!
  TBranch        *b_t_photonEne25x25;   //!
  TBranch        *b_t_eleEne25x25;   //!
  TBranch        *b_t_muEne25x25;   //!
  TBranch        *b_t_maxNearP21x21;   //!
  TBranch        *b_t_cHadronEne21x21;   //!
  TBranch        *b_t_cHadronEne21x21_1;   //!
  TBranch        *b_t_cHadronEne21x21_2;   //!
  TBranch        *b_t_cHadronEne21x21_3;   //!
  TBranch        *b_t_nHadronEne21x21;   //!
  TBranch        *b_t_photonEne21x21;   //!
  TBranch        *b_t_eleEne21x21;   //!
  TBranch        *b_t_muEne21x21;   //!
  TBranch        *b_t_maxNearP15x15;   //!
  TBranch        *b_t_cHadronEne15x15;   //!
  TBranch        *b_t_cHadronEne15x15_1;   //!
  TBranch        *b_t_cHadronEne15x15_2;   //!
  TBranch        *b_t_cHadronEne15x15_3;   //!
  TBranch        *b_t_nHadronEne15x15;   //!
  TBranch        *b_t_photonEne15x15;   //!
  TBranch        *b_t_eleEne15x15;   //!
  TBranch        *b_t_muEne15x15;   //!
  TBranch        *b_t_maxNearP11x11;   //!
  TBranch        *b_t_cHadronEne11x11;   //!
  TBranch        *b_t_cHadronEne11x11_1;   //!
  TBranch        *b_t_cHadronEne11x11_2;   //!
  TBranch        *b_t_cHadronEne11x11_3;   //!
  TBranch        *b_t_nHadronEne11x11;   //!
  TBranch        *b_t_photonEne11x11;   //!
  TBranch        *b_t_eleEne11x11;   //!
  TBranch        *b_t_muEne11x11;   //!
  TBranch        *b_t_maxNearP9x9;   //!
  TBranch        *b_t_cHadronEne9x9;   //!
  TBranch        *b_t_cHadronEne9x9_1;   //!
  TBranch        *b_t_cHadronEne9x9_2;   //!
  TBranch        *b_t_cHadronEne9x9_3;   //!
  TBranch        *b_t_nHadronEne9x9;   //!
  TBranch        *b_t_photonEne9x9;   //!
  TBranch        *b_t_eleEne9x9;   //!
  TBranch        *b_t_muEne9x9;   //!
  TBranch        *b_t_maxNearP7x7;   //!
  TBranch        *b_t_cHadronEne7x7;   //!
  TBranch        *b_t_cHadronEne7x7_1;   //!
  TBranch        *b_t_cHadronEne7x7_2;   //!
  TBranch        *b_t_cHadronEne7x7_3;   //!
  TBranch        *b_t_nHadronEne7x7;   //!
  TBranch        *b_t_photonEne7x7;   //!
  TBranch        *b_t_eleEne7x7;   //!
  TBranch        *b_t_muEne7x7;   //!
  TBranch        *b_t_maxNearPHC3x3;   //!
  TBranch        *b_t_cHadronEneHC3x3;   //!
  TBranch        *b_t_cHadronEneHC3x3_1;   //!
  TBranch        *b_t_cHadronEneHC3x3_2;   //!
  TBranch        *b_t_cHadronEneHC3x3_3;   //!
  TBranch        *b_t_nHadronEneHC3x3;   //!
  TBranch        *b_t_photonEneHC3x3;   //!
  TBranch        *b_t_eleEneHC3x3;   //!
  TBranch        *b_t_muEneHC3x3;   //!
  TBranch        *b_t_maxNearPHC5x5;   //!
  TBranch        *b_t_cHadronEneHC5x5;   //!
  TBranch        *b_t_cHadronEneHC5x5_1;   //!
  TBranch        *b_t_cHadronEneHC5x5_2;   //!
  TBranch        *b_t_cHadronEneHC5x5_3;   //!
  TBranch        *b_t_nHadronEneHC5x5;   //!
  TBranch        *b_t_photonEneHC5x5;   //!
  TBranch        *b_t_eleEneHC5x5;   //!
  TBranch        *b_t_muEneHC5x5;   //!
  TBranch        *b_t_maxNearPHC7x7;   //!
  TBranch        *b_t_cHadronEneHC7x7;   //!
  TBranch        *b_t_cHadronEneHC7x7_1;   //!
  TBranch        *b_t_cHadronEneHC7x7_2;   //!
  TBranch        *b_t_cHadronEneHC7x7_3;   //!
  TBranch        *b_t_nHadronEneHC7x7;   //!
  TBranch        *b_t_photonEneHC7x7;   //!
  TBranch        *b_t_eleEneHC7x7;   //!
  TBranch        *b_t_muEneHC7x7;   //!
  TBranch        *b_t_maxNearPR;   //!
  TBranch        *b_t_cHadronEneR;   //!
  TBranch        *b_t_cHadronEneR_1;   //!
  TBranch        *b_t_cHadronEneR_2;   //!
  TBranch        *b_t_cHadronEneR_3;   //!
  TBranch        *b_t_nHadronEneR;   //!
  TBranch        *b_t_photonEneR;   //!
  TBranch        *b_t_eleEneR;   //!
  TBranch        *b_t_muEneR;   //!
  TBranch        *b_t_maxNearPIsoR;   //!
  TBranch        *b_t_cHadronEneIsoR;   //!
  TBranch        *b_t_cHadronEneIsoR_1;   //!
  TBranch        *b_t_cHadronEneIsoR_2;   //!
  TBranch        *b_t_cHadronEneIsoR_3;   //!
  TBranch        *b_t_nHadronEneIsoR;   //!
  TBranch        *b_t_photonEneIsoR;   //!
  TBranch        *b_t_eleEneIsoR;   //!
  TBranch        *b_t_muEneIsoR;   //!
  TBranch        *b_t_maxNearPHCR;   //!
  TBranch        *b_t_cHadronEneHCR;   //!
  TBranch        *b_t_cHadronEneHCR_1;   //!
  TBranch        *b_t_cHadronEneHCR_2;   //!
  TBranch        *b_t_cHadronEneHCR_3;   //!
  TBranch        *b_t_nHadronEneHCR;   //!
  TBranch        *b_t_photonEneHCR;   //!
  TBranch        *b_t_eleEneHCR;   //!
  TBranch        *b_t_muEneHCR;   //!
  TBranch        *b_t_maxNearPIsoHCR;   //!
  TBranch        *b_t_cHadronEneIsoHCR;   //!
  TBranch        *b_t_cHadronEneIsoHCR_1;   //!
  TBranch        *b_t_cHadronEneIsoHCR_2;   //!
  TBranch        *b_t_cHadronEneIsoHCR_3;   //!
  TBranch        *b_t_nHadronEneIsoHCR;   //!
  TBranch        *b_t_photonEneIsoHCR;   //!
  TBranch        *b_t_eleEneIsoHCR;   //!
  TBranch        *b_t_muEneIsoHCR;   //!
  TBranch        *b_t_L1Decision;   //!
  TBranch        *b_t_L1CenJetPt;   //!
  TBranch        *b_t_L1CenJetEta;   //!
  TBranch        *b_t_L1CenJetPhi;   //!
  TBranch        *b_t_L1FwdJetPt;   //!
  TBranch        *b_t_L1FwdJetEta;   //!
  TBranch        *b_t_L1FwdJetPhi;   //!
  TBranch        *b_t_L1TauJetPt;   //!
  TBranch        *b_t_L1TauJetEta;   //!
  TBranch        *b_t_L1TauJetPhi;   //!
  TBranch        *b_t_L1MuonPt;   //!
  TBranch        *b_t_L1MuonEta;   //!
  TBranch        *b_t_L1MuonPhi;   //!
  TBranch        *b_t_L1IsoEMPt;   //!
  TBranch        *b_t_L1IsoEMEta;   //!
  TBranch        *b_t_L1IsoEMPhi;   //!
  TBranch        *b_t_L1NonIsoEMPt;   //!
  TBranch        *b_t_L1NonIsoEMEta;   //!
  TBranch        *b_t_L1NonIsoEMPhi;   //!
  TBranch        *b_t_L1METPt;   //!
  TBranch        *b_t_L1METEta;   //!
  TBranch        *b_t_L1METPhi;   //!
  
  TreeAnalysisReadGen(TChain *tree, const char *outFileName);
  virtual ~TreeAnalysisReadGen();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TChain *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  void             getL1Names();
  void             BookHistograms(const char *outFileName);
  double           DeltaPhi(double v1, double v2);
  double           DeltaR(double eta1, double phi1, double eta2, double phi2);
};

#endif
