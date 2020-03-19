//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Oct 25 18:21:22 2019 by ROOT version 6.14/09
// from TTree HBHEMuonHighEta/HBHEMuonHighEta
// found on file: muonHighEta.root
//////////////////////////////////////////////////////////
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TTree.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

class HBHEMuonHighEta {

public :
  HBHEMuonHighEta(const char *infile, const char *outfile,
		  const int mode=1, const bool debug=false);
  virtual ~HBHEMuonHighEta();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

private:
  void             BookHistograms(const char* fname);
  void             Close();

  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Fixed size dimensions of array or collections stored in the TTree if any.
  // Declaration of leaf types
  std::vector<double>       *pt_of_muon;
  std::vector<double>       *eta_of_muon;
  std::vector<double>       *phi_of_muon;
  std::vector<double>       *energy_of_muon;
  std::vector<double>       *p_of_muon;
  std::vector<bool>         *MediumMuon;
  std::vector<double>       *IsolationR04;
  std::vector<double>       *IsolationR03;
  std::vector<double>       *ecal_3into3;
  std::vector<double>       *hcal_3into3;
  std::vector<double>       *ho_3into3;
  std::vector<double>       *emaxNearP;
  UInt_t                     Run_No;
  UInt_t                     Event_No;
  UInt_t                     GoodVertex;
  std::vector<bool>         *matchedId;
  std::vector<bool>         *hcal_cellHot;
  std::vector<double>       *ecal_3x3;
  std::vector<double>       *hcal_1x1;
  std::vector<unsigned int> *ecal_detID;
  std::vector<unsigned int> *hcal_detID;
  std::vector<unsigned int> *ehcal_detID;
  std::vector<int>          *hcal_ieta;
  std::vector<int>          *hcal_iphi;
  std::vector<double>       *hcal_edepth1;
  std::vector<double>       *hcal_activeL1;
  std::vector<double>       *hcal_edepthHot1;
  std::vector<double>       *hcal_activeHotL1;
  std::vector<double>       *hcal_cdepthHot1;
  std::vector<double>       *hcal_cdepthHotBG1;
  std::vector<double>       *hcal_edepthCorrect1;
  std::vector<double>       *hcal_edepthHotCorrect1;
  std::vector<bool>         *hcal_depthMatch1;
  std::vector<bool>         *hcal_depthMatchHot1;
  std::vector<double>       *hcal_edepth2;
  std::vector<double>       *hcal_activeL2;
  std::vector<double>       *hcal_edepthHot2;
  std::vector<double>       *hcal_activeHotL2;
  std::vector<double>       *hcal_cdepthHot2;
  std::vector<double>       *hcal_cdepthHotBG2;
  std::vector<double>       *hcal_edepthCorrect2;
  std::vector<double>       *hcal_edepthHotCorrect2;
  std::vector<bool>         *hcal_depthMatch2;
  std::vector<bool>         *hcal_depthMatchHot2;
  std::vector<double>       *hcal_edepth3;
  std::vector<double>       *hcal_activeL3;
  std::vector<double>       *hcal_edepthHot3;
  std::vector<double>       *hcal_activeHotL3;
  std::vector<double>       *hcal_cdepthHot3;
  std::vector<double>       *hcal_cdepthHotBG3;
  std::vector<double>       *hcal_edepthCorrect3;
  std::vector<double>       *hcal_edepthHotCorrect3;
  std::vector<bool>         *hcal_depthMatch3;
  std::vector<bool>         *hcal_depthMatchHot3;
  std::vector<double>       *hcal_edepth4;
  std::vector<double>       *hcal_activeL4;
  std::vector<double>       *hcal_edepthHot4;
  std::vector<double>       *hcal_activeHotL4;
  std::vector<double>       *hcal_cdepthHot4;
  std::vector<double>       *hcal_cdepthHotBG4;
  std::vector<double>       *hcal_edepthCorrect4;
  std::vector<double>       *hcal_edepthHotCorrect4;
  std::vector<bool>         *hcal_depthMatch4;
  std::vector<bool>         *hcal_depthMatchHot4;
  std::vector<double>       *hcal_edepth5;
  std::vector<double>       *hcal_activeL5;
  std::vector<double>       *hcal_edepthHot5;
  std::vector<double>       *hcal_activeHotL5;
  std::vector<double>       *hcal_cdepthHot5;
  std::vector<double>       *hcal_cdepthHotBG5;
  std::vector<double>       *hcal_edepthCorrect5;
  std::vector<double>       *hcal_edepthHotCorrect5;
  std::vector<bool>         *hcal_depthMatch5;
  std::vector<bool>         *hcal_depthMatchHot5;
  std::vector<double>       *hcal_edepth6;
  std::vector<double>       *hcal_activeL6;
  std::vector<double>       *hcal_edepthHot6;
  std::vector<double>       *hcal_activeHotL6;
  std::vector<double>       *hcal_cdepthHot6;
  std::vector<double>       *hcal_cdepthHotBG6;
  std::vector<double>       *hcal_edepthCorrect6;
  std::vector<double>       *hcal_edepthHotCorrect6;
  std::vector<bool>         *hcal_depthMatch6;
  std::vector<bool>         *hcal_depthMatchHot6;
  std::vector<double>       *hcal_edepth7;
  std::vector<double>       *hcal_activeL7;
  std::vector<double>       *hcal_edepthHot7;
  std::vector<double>       *hcal_activeHotL7;
  std::vector<double>       *hcal_cdepthHot7;
  std::vector<double>       *hcal_cdepthHotBG7;
  std::vector<double>       *hcal_edepthCorrect7;
  std::vector<double>       *hcal_edepthHotCorrect7;
  std::vector<bool>         *hcal_depthMatch7;
  std::vector<bool>         *hcal_depthMatchHot7;
  std::vector<double>       *activeLength;
  std::vector<double>       *activeLengthHot;
  std::vector<double>       *trackDz;
  std::vector<int>          *trackLayerCrossed;
  std::vector<int>          *trackOuterHit;
  std::vector<int>          *trackMissedInnerHits;
  std::vector<int>          *trackMissedOuterHits;

  // List of branches
  TBranch                   *b_pt_of_muon;
  TBranch                   *b_eta_of_muon;
  TBranch                   *b_phi_of_muon;
  TBranch                   *b_energy_of_muon;
  TBranch                   *b_p_of_muon;
  TBranch                   *b_MediumMuon;
  TBranch                   *b_IsolationR04;
  TBranch                   *b_IsolationR03;
  TBranch                   *b_ecal_3into3;
  TBranch                   *b_hcal_3into3;
  TBranch                   *b_ho_3into3;
  TBranch                   *b_emaxNearP;
  TBranch                   *b_Run_No;
  TBranch                   *b_Event_No;
  TBranch                   *b_GoodVertex;
  TBranch                   *b_matchedId;
  TBranch                   *b_hcal_cellHot;
  TBranch                   *b_ecal_3x3;
  TBranch                   *b_hcal_1x1;
  TBranch                   *b_ecal_detID;
  TBranch                   *b_hcal_detID;
  TBranch                   *b_ehcal_detID;
  TBranch                   *b_hcal_ieta;
  TBranch                   *b_hcal_iphi;
  TBranch                   *b_hcal_edepth1;
  TBranch                   *b_hcal_activeL1;
  TBranch                   *b_hcal_edepthHot1;
  TBranch                   *b_hcal_activeHotL1;
  TBranch                   *b_hcal_cdepthHot1;
  TBranch                   *b_hcal_cdepthHotBG1;
  TBranch                   *b_hcal_edepthCorrect1;
  TBranch                   *b_hcal_edepthHotCorrect1;
  TBranch                   *b_hcal_depthMatch1;
  TBranch                   *b_hcal_depthMatchHot1;
  TBranch                   *b_hcal_edepth2;
  TBranch                   *b_hcal_activeL2;
  TBranch                   *b_hcal_edepthHot2;
  TBranch                   *b_hcal_activeHotL2;
  TBranch                   *b_hcal_cdepthHot2;
  TBranch                   *b_hcal_cdepthHotBG2;
  TBranch                   *b_hcal_edepthCorrect2;
  TBranch                   *b_hcal_edepthHotCorrect2;
  TBranch                   *b_hcal_depthMatch2;
  TBranch                   *b_hcal_depthMatchHot2;
  TBranch                   *b_hcal_edepth3;
  TBranch                   *b_hcal_activeL3;
  TBranch                   *b_hcal_edepthHot3;
  TBranch                   *b_hcal_activeHotL3;
  TBranch                   *b_hcal_cdepthHot3;
  TBranch                   *b_hcal_cdepthHotBG3;
  TBranch                   *b_hcal_edepthCorrect3;
  TBranch                   *b_hcal_edepthHotCorrect3;
  TBranch                   *b_hcal_depthMatch3;
  TBranch                   *b_hcal_depthMatchHot3;
  TBranch                   *b_hcal_edepth4;
  TBranch                   *b_hcal_activeL4;
  TBranch                   *b_hcal_edepthHot4;
  TBranch                   *b_hcal_activeHotL4;
  TBranch                   *b_hcal_cdepthHot4;
  TBranch                   *b_hcal_cdepthHotBG4;
  TBranch                   *b_hcal_edepthCorrect4;
  TBranch                   *b_hcal_edepthHotCorrect4;
  TBranch                   *b_hcal_depthMatch4;
  TBranch                   *b_hcal_depthMatchHot4;
  TBranch                   *b_hcal_edepth5;
  TBranch                   *b_hcal_activeL5;
  TBranch                   *b_hcal_edepthHot5;
  TBranch                   *b_hcal_activeHotL5;
  TBranch                   *b_hcal_cdepthHot5;
  TBranch                   *b_hcal_cdepthHotBG5;
  TBranch                   *b_hcal_edepthCorrect5;
  TBranch                   *b_hcal_edepthHotCorrect5;
  TBranch                   *b_hcal_depthMatch5;
  TBranch                   *b_hcal_depthMatchHot5;
  TBranch                   *b_hcal_edepth6;
  TBranch                   *b_hcal_activeL6;
  TBranch                   *b_hcal_edepthHot6;
  TBranch                   *b_hcal_activeHotL6;
  TBranch                   *b_hcal_cdepthHot6;
  TBranch                   *b_hcal_cdepthHotBG6;
  TBranch                   *b_hcal_edepthCorrect6;
  TBranch                   *b_hcal_edepthHotCorrect6;
  TBranch                   *b_hcal_depthMatch6;
  TBranch                   *b_hcal_depthMatchHot6;
  TBranch                   *b_hcal_edepth7;
  TBranch                   *b_hcal_activeL7;
  TBranch                   *b_hcal_edepthHot7;
  TBranch                   *b_hcal_activeHotL7;
  TBranch                   *b_hcal_cdepthHot7;
  TBranch                   *b_hcal_cdepthHotBG7;
  TBranch                   *b_hcal_edepthCorrect7;
  TBranch                   *b_hcal_edepthHotCorrect7;
  TBranch                   *b_hcal_depthMatch7;
  TBranch                   *b_hcal_depthMatchHot7;
  TBranch                   *b_activeLength;
  TBranch                   *b_activeLengthHot;
  TBranch                   *b_trackDz;
  TBranch                   *b_trackLayerCrossed;
  TBranch                   *b_trackOuterHit;
  TBranch                   *b_trackMissedInnerHits;
  TBranch                   *b_trackMissedOuterHits;

  int                        modeLHC_;
  bool                       debug_;
  TFile                     *output_file;
};

HBHEMuonHighEta::HBHEMuonHighEta(const char *infile, const char *outfile,
				 const int mode, const bool debug) {
  modeLHC_ = mode;
  debug_ = debug;
  TFile      *file = new TFile(infile);
  TDirectory *dir  = (TDirectory*)(file->FindObjectAny("hcalHBHEMuonHighEta"));
  TTree      *tree = (TTree*)(dir->FindObjectAny("HBHEMuonHighEta"));
  std::cout << "Attaches tree HBHEMuonHighEta at " << tree << " in file " 
	    << infile << std::endl;
  
  BookHistograms(outfile);
  Init(tree);
}

HBHEMuonHighEta::~HBHEMuonHighEta() {
  Close();
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t HBHEMuonHighEta::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t HBHEMuonHighEta::LoadTree(Long64_t entry) {
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

void HBHEMuonHighEta::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  pt_of_muon = 0;
  eta_of_muon = 0;
  phi_of_muon = 0;
  energy_of_muon = 0;
  p_of_muon = 0;
  MediumMuon = 0;
  IsolationR04 = 0;
  IsolationR03 = 0;
  ecal_3into3 = 0;
  hcal_3into3 = 0;
  ho_3into3 = 0;
  emaxNearP = 0;
  matchedId = 0;
  hcal_cellHot = 0;
  ecal_3x3 = 0;
  hcal_1x1 = 0;
  ecal_detID = 0;
  hcal_detID = 0;
  ehcal_detID = 0;
  hcal_ieta = 0;
  hcal_iphi = 0;
  hcal_edepth1 = 0;
  hcal_activeL1 = 0;
  hcal_edepthHot1 = 0;
  hcal_activeHotL1 = 0;
  hcal_cdepthHot1 = 0;
  hcal_cdepthHotBG1 = 0;
  hcal_edepthCorrect1 = 0;
  hcal_edepthHotCorrect1 = 0;
  hcal_depthMatch1 = 0;
  hcal_depthMatchHot1 = 0;
  hcal_edepth2 = 0;
  hcal_activeL2 = 0;
  hcal_edepthHot2 = 0;
  hcal_activeHotL2 = 0;
  hcal_cdepthHot2 = 0;
  hcal_cdepthHotBG2 = 0;
  hcal_edepthCorrect2 = 0;
  hcal_edepthHotCorrect2 = 0;
  hcal_depthMatch2 = 0;
  hcal_depthMatchHot2 = 0;
  hcal_edepth3 = 0;
  hcal_activeL3 = 0;
  hcal_edepthHot3 = 0;
  hcal_activeHotL3 = 0;
  hcal_cdepthHot3 = 0;
  hcal_cdepthHotBG3 = 0;
  hcal_edepthCorrect3 = 0;
  hcal_edepthHotCorrect3 = 0;
  hcal_depthMatch3 = 0;
  hcal_depthMatchHot3 = 0;
  hcal_edepth4 = 0;
  hcal_activeL4 = 0;
  hcal_edepthHot4 = 0;
  hcal_activeHotL4 = 0;
  hcal_cdepthHot4 = 0;
  hcal_cdepthHotBG4 = 0;
  hcal_edepthCorrect4 = 0;
  hcal_edepthHotCorrect4 = 0;
  hcal_depthMatch4 = 0;
  hcal_depthMatchHot4 = 0;
  hcal_edepth5 = 0;
  hcal_activeL5 = 0;
  hcal_edepthHot5 = 0;
  hcal_activeHotL5 = 0;
  hcal_cdepthHot5 = 0;
  hcal_cdepthHotBG5 = 0;
  hcal_edepthCorrect5 = 0;
  hcal_edepthHotCorrect5 = 0;
  hcal_depthMatch5 = 0;
  hcal_depthMatchHot5 = 0;
  hcal_edepth6 = 0;
  hcal_activeL6 = 0;
  hcal_edepthHot6 = 0;
  hcal_activeHotL6 = 0;
  hcal_cdepthHot6 = 0;
  hcal_cdepthHotBG6 = 0;
  hcal_edepthCorrect6 = 0;
  hcal_edepthHotCorrect6 = 0;
  hcal_depthMatch6 = 0;
  hcal_depthMatchHot6 = 0;
  hcal_edepth7 = 0;
  hcal_activeL7 = 0;
  hcal_edepthHot7 = 0;
  hcal_activeHotL7 = 0;
  hcal_cdepthHot7 = 0;
  hcal_cdepthHotBG7 = 0;
  hcal_edepthCorrect7 = 0;
  hcal_edepthHotCorrect7 = 0;
  hcal_depthMatch7 = 0;
  hcal_depthMatchHot7 = 0;
  activeLength = 0;
  activeLengthHot = 0;
  trackDz = 0;
  trackLayerCrossed = 0;
  trackOuterHit = 0;
  trackMissedInnerHits = 0;
  trackMissedOuterHits = 0;
  
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("pt_of_muon", &pt_of_muon, &b_pt_of_muon);
  fChain->SetBranchAddress("eta_of_muon", &eta_of_muon, &b_eta_of_muon);
  fChain->SetBranchAddress("phi_of_muon", &phi_of_muon, &b_phi_of_muon);
  fChain->SetBranchAddress("energy_of_muon", &energy_of_muon, &b_energy_of_muon);
  fChain->SetBranchAddress("p_of_muon", &p_of_muon, &b_p_of_muon);
  fChain->SetBranchAddress("MediumMuon", &MediumMuon, &b_MediumMuon);
  fChain->SetBranchAddress("IsolationR04", &IsolationR04, &b_IsolationR04);
  fChain->SetBranchAddress("IsolationR03", &IsolationR03, &b_IsolationR03);
  fChain->SetBranchAddress("ecal_3into3", &ecal_3into3, &b_ecal_3into3);
  fChain->SetBranchAddress("hcal_3into3", &hcal_3into3, &b_hcal_3into3);
  fChain->SetBranchAddress("ho_3into3", &ho_3into3, &b_ho_3into3);
  fChain->SetBranchAddress("emaxNearP", &emaxNearP, &b_emaxNearP);
  fChain->SetBranchAddress("Run_No", &Run_No, &b_Run_No);
  fChain->SetBranchAddress("Event_No", &Event_No, &b_Event_No);
  fChain->SetBranchAddress("GoodVertex", &GoodVertex, &b_GoodVertex);
  fChain->SetBranchAddress("matchedId", &matchedId, &b_matchedId);
  fChain->SetBranchAddress("hcal_cellHot", &hcal_cellHot, &b_hcal_cellHot);
  fChain->SetBranchAddress("ecal_3x3", &ecal_3x3, &b_ecal_3x3);
  fChain->SetBranchAddress("hcal_1x1", &hcal_1x1, &b_hcal_1x1);
  fChain->SetBranchAddress("ecal_detID", &ecal_detID, &b_ecal_detID);
  fChain->SetBranchAddress("hcal_detID", &hcal_detID, &b_hcal_detID);
  fChain->SetBranchAddress("ehcal_detID", &ehcal_detID, &b_ehcal_detID);
  fChain->SetBranchAddress("hcal_ieta", &hcal_ieta, &b_hcal_ieta);
  fChain->SetBranchAddress("hcal_iphi", &hcal_iphi, &b_hcal_iphi);
  fChain->SetBranchAddress("hcal_edepth1", &hcal_edepth1, &b_hcal_edepth1);
  fChain->SetBranchAddress("hcal_activeL1", &hcal_activeL1, &b_hcal_activeL1);
  fChain->SetBranchAddress("hcal_edepthHot1", &hcal_edepthHot1, &b_hcal_edepthHot1);
  fChain->SetBranchAddress("hcal_activeHotL1", &hcal_activeHotL1, &b_hcal_activeHotL1);
  fChain->SetBranchAddress("hcal_cdepthHot1", &hcal_cdepthHot1, &b_hcal_cdepthHot1);
  fChain->SetBranchAddress("hcal_cdepthHotBG1", &hcal_cdepthHotBG1, &b_hcal_cdepthHotBG1);
  fChain->SetBranchAddress("hcal_edepthCorrect1", &hcal_edepthCorrect1, &b_hcal_edepthCorrect1);
  fChain->SetBranchAddress("hcal_edepthHotCorrect1", &hcal_edepthHotCorrect1, &b_hcal_edepthHotCorrect1);
  fChain->SetBranchAddress("hcal_depthMatch1", &hcal_depthMatch1, &b_hcal_depthMatch1);
  fChain->SetBranchAddress("hcal_depthMatchHot1", &hcal_depthMatchHot1, &b_hcal_depthMatchHot1);
  fChain->SetBranchAddress("hcal_edepth2", &hcal_edepth2, &b_hcal_edepth2);
  fChain->SetBranchAddress("hcal_activeL2", &hcal_activeL2, &b_hcal_activeL2);
  fChain->SetBranchAddress("hcal_edepthHot2", &hcal_edepthHot2, &b_hcal_edepthHot2);
  fChain->SetBranchAddress("hcal_activeHotL2", &hcal_activeHotL2, &b_hcal_activeHotL2);
  fChain->SetBranchAddress("hcal_cdepthHot2", &hcal_cdepthHot2, &b_hcal_cdepthHot2);
  fChain->SetBranchAddress("hcal_cdepthHotBG2", &hcal_cdepthHotBG2, &b_hcal_cdepthHotBG2);
  fChain->SetBranchAddress("hcal_edepthCorrect2", &hcal_edepthCorrect2, &b_hcal_edepthCorrect2);
  fChain->SetBranchAddress("hcal_edepthHotCorrect2", &hcal_edepthHotCorrect2, &b_hcal_edepthHotCorrect2);
  fChain->SetBranchAddress("hcal_depthMatch2", &hcal_depthMatch2, &b_hcal_depthMatch2);
  fChain->SetBranchAddress("hcal_depthMatchHot2", &hcal_depthMatchHot2, &b_hcal_depthMatchHot2);
  fChain->SetBranchAddress("hcal_edepth3", &hcal_edepth3, &b_hcal_edepth3);
  fChain->SetBranchAddress("hcal_activeL3", &hcal_activeL3, &b_hcal_activeL3);
  fChain->SetBranchAddress("hcal_edepthHot3", &hcal_edepthHot3, &b_hcal_edepthHot3);
  fChain->SetBranchAddress("hcal_activeHotL3", &hcal_activeHotL3, &b_hcal_activeHotL3);
  fChain->SetBranchAddress("hcal_cdepthHot3", &hcal_cdepthHot3, &b_hcal_cdepthHot3);
  fChain->SetBranchAddress("hcal_cdepthHotBG3", &hcal_cdepthHotBG3, &b_hcal_cdepthHotBG3);
  fChain->SetBranchAddress("hcal_edepthCorrect3", &hcal_edepthCorrect3, &b_hcal_edepthCorrect3);
  fChain->SetBranchAddress("hcal_edepthHotCorrect3", &hcal_edepthHotCorrect3, &b_hcal_edepthHotCorrect3);
  fChain->SetBranchAddress("hcal_depthMatch3", &hcal_depthMatch3, &b_hcal_depthMatch3);
  fChain->SetBranchAddress("hcal_depthMatchHot3", &hcal_depthMatchHot3, &b_hcal_depthMatchHot3);
  fChain->SetBranchAddress("hcal_edepth4", &hcal_edepth4, &b_hcal_edepth4);
  fChain->SetBranchAddress("hcal_activeL4", &hcal_activeL4, &b_hcal_activeL4);
  fChain->SetBranchAddress("hcal_edepthHot4", &hcal_edepthHot4, &b_hcal_edepthHot4);
  fChain->SetBranchAddress("hcal_activeHotL4", &hcal_activeHotL4, &b_hcal_activeHotL4);
  fChain->SetBranchAddress("hcal_cdepthHot4", &hcal_cdepthHot4, &b_hcal_cdepthHot4);
  fChain->SetBranchAddress("hcal_cdepthHotBG4", &hcal_cdepthHotBG4, &b_hcal_cdepthHotBG4);
  fChain->SetBranchAddress("hcal_edepthCorrect4", &hcal_edepthCorrect4, &b_hcal_edepthCorrect4);
  fChain->SetBranchAddress("hcal_edepthHotCorrect4", &hcal_edepthHotCorrect4, &b_hcal_edepthHotCorrect4);
  fChain->SetBranchAddress("hcal_depthMatch4", &hcal_depthMatch4, &b_hcal_depthMatch4);
  fChain->SetBranchAddress("hcal_depthMatchHot4", &hcal_depthMatchHot4, &b_hcal_depthMatchHot4);
  fChain->SetBranchAddress("hcal_edepth5", &hcal_edepth5, &b_hcal_edepth5);
  fChain->SetBranchAddress("hcal_activeL5", &hcal_activeL5, &b_hcal_activeL5);
  fChain->SetBranchAddress("hcal_edepthHot5", &hcal_edepthHot5, &b_hcal_edepthHot5);
  fChain->SetBranchAddress("hcal_activeHotL5", &hcal_activeHotL5, &b_hcal_activeHotL5);
  fChain->SetBranchAddress("hcal_cdepthHot5", &hcal_cdepthHot5, &b_hcal_cdepthHot5);
  fChain->SetBranchAddress("hcal_cdepthHotBG5", &hcal_cdepthHotBG5, &b_hcal_cdepthHotBG5);
  fChain->SetBranchAddress("hcal_edepthCorrect5", &hcal_edepthCorrect5, &b_hcal_edepthCorrect5);
  fChain->SetBranchAddress("hcal_edepthHotCorrect5", &hcal_edepthHotCorrect5, &b_hcal_edepthHotCorrect5);
  fChain->SetBranchAddress("hcal_depthMatch5", &hcal_depthMatch5, &b_hcal_depthMatch5);
  fChain->SetBranchAddress("hcal_depthMatchHot5", &hcal_depthMatchHot5, &b_hcal_depthMatchHot5);
  fChain->SetBranchAddress("hcal_edepth6", &hcal_edepth6, &b_hcal_edepth6);
  fChain->SetBranchAddress("hcal_activeL6", &hcal_activeL6, &b_hcal_activeL6);
  fChain->SetBranchAddress("hcal_edepthHot6", &hcal_edepthHot6, &b_hcal_edepthHot6);
  fChain->SetBranchAddress("hcal_activeHotL6", &hcal_activeHotL6, &b_hcal_activeHotL6);
  fChain->SetBranchAddress("hcal_cdepthHot6", &hcal_cdepthHot6, &b_hcal_cdepthHot6);
  fChain->SetBranchAddress("hcal_cdepthHotBG6", &hcal_cdepthHotBG6, &b_hcal_cdepthHotBG6);
  fChain->SetBranchAddress("hcal_edepthCorrect6", &hcal_edepthCorrect6, &b_hcal_edepthCorrect6);
  fChain->SetBranchAddress("hcal_edepthHotCorrect6", &hcal_edepthHotCorrect6, &b_hcal_edepthHotCorrect6);
  fChain->SetBranchAddress("hcal_depthMatch6", &hcal_depthMatch6, &b_hcal_depthMatch6);
  fChain->SetBranchAddress("hcal_depthMatchHot6", &hcal_depthMatchHot6, &b_hcal_depthMatchHot6);
  fChain->SetBranchAddress("hcal_edepth7", &hcal_edepth7, &b_hcal_edepth7);
  fChain->SetBranchAddress("hcal_activeL7", &hcal_activeL7, &b_hcal_activeL7);
  fChain->SetBranchAddress("hcal_edepthHot7", &hcal_edepthHot7, &b_hcal_edepthHot7);
  fChain->SetBranchAddress("hcal_activeHotL7", &hcal_activeHotL7, &b_hcal_activeHotL7);
  fChain->SetBranchAddress("hcal_cdepthHot7", &hcal_cdepthHot7, &b_hcal_cdepthHot7);
  fChain->SetBranchAddress("hcal_cdepthHotBG7", &hcal_cdepthHotBG7, &b_hcal_cdepthHotBG7);
  fChain->SetBranchAddress("hcal_edepthCorrect7", &hcal_edepthCorrect7, &b_hcal_edepthCorrect7);
  fChain->SetBranchAddress("hcal_edepthHotCorrect7", &hcal_edepthHotCorrect7, &b_hcal_edepthHotCorrect7);
  fChain->SetBranchAddress("hcal_depthMatch7", &hcal_depthMatch7, &b_hcal_depthMatch7);
  fChain->SetBranchAddress("hcal_depthMatchHot7", &hcal_depthMatchHot7, &b_hcal_depthMatchHot7);
  fChain->SetBranchAddress("activeLength", &activeLength, &b_activeLength);
  fChain->SetBranchAddress("activeLengthHot", &activeLengthHot, &b_activeLengthHot);
  fChain->SetBranchAddress("trackDz", &trackDz, &b_trackDz);
  fChain->SetBranchAddress("trackLayerCrossed", &trackLayerCrossed, &b_trackLayerCrossed);
  fChain->SetBranchAddress("trackOuterHit", &trackOuterHit, &b_trackOuterHit);
  fChain->SetBranchAddress("trackMissedInnerHits", &trackMissedInnerHits, &b_trackMissedInnerHits);
  fChain->SetBranchAddress("trackMissedOuterHits", &trackMissedOuterHits, &b_trackMissedOuterHits);
  Notify();
}

Bool_t HBHEMuonHighEta::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
   return kTRUE;
}

void HBHEMuonHighEta::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t HBHEMuonHighEta::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void HBHEMuonHighEta::Loop() {
  //   In a ROOT session, you can do:
  //      root> .L HBHEMuonHighEta.C+g
  //      root> HBHEMuonHighEta t
  //      root> t.GetEntry(12); // Fill t data members with entry number 12
  //      root> t.Show();       // Show values of entry 12
  //      root> t.Show(16);     // Read and show values of entry 16
  //      root> t.Loop();       // Loop on all entries
  //
  
  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
   }
}

void HBHEMuonHighEta::BookHistograms(const char* fname) {

  output_file = TFile::Open(fname,"RECREATE");
}

void HBHEMuonHighEta::Close() {
  output_file->cd();
  if (debug_) std::cout << "file yet to be Written" << std::endl;
  output_file->Write();
  std::cout << "output file Written" << std::endl;
  output_file->Close();
  if (debug_) std::cout << "now doing return" << std::endl;
}
