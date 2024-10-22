///////////////////////////////////////////////////////////////////////////////
//
//   HBHEMuonOfflineSimAnalyzer h1(infile, outfile, mode, maxDHB, maxDHE);
//   h1.Loop()
//
//      Offline analysis for MC files
//
//   infile     const char*  Name of the input file
//   outfile    const char*  Name of the output file
//                           (dyll_PU20_25_output.root)
//   mode       int          Geometry file used 0:(defined by maxDHB/HE);
//                           1 (Run 1; valid till 2016); 2 (Run 2; 2018);
//                           3 (Run 3; post LS2); 4 (2017 Plan 1);
//                           5 (Run 4; post LS3); default (2)
//   maxDHB     int          Maximum number of depths for HB (4)
//   maxDHE     int          Maximum number of depths for HE (7)
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>

class HBHEMuonOfflineSimAnalyzer {
private:
  TTree *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t fCurrent;

  UInt_t Run_No;
  UInt_t Event_No;
  UInt_t LumiNumber;
  UInt_t BXNumber;
  double pt_of_muon;
  double eta_of_muon;
  double phi_of_muon;
  double p_of_muon;
  double ecal_3x3;
  unsigned int ecal_detID;
  double hcal_1x1;
  double matchedId;
  unsigned int hcal_detID;
  unsigned int hcal_cellHot;
  double activeLength;
  double hcal_edepth1;
  double hcal_edepth2;
  double hcal_edepth3;
  double hcal_edepth4;
  double hcal_activeL1;
  double hcal_activeL2;
  double hcal_activeL3;
  double hcal_activeL4;
  double activeLengthHot;
  double hcal_edepthHot1;
  double hcal_edepthHot2;
  double hcal_edepthHot3;
  double hcal_edepthHot4;
  double hcal_activeHotL1;
  double hcal_activeHotL2;
  double hcal_activeHotL3;
  double hcal_activeHotL4;
  double hcal_edepth5;
  double hcal_activeL5;
  double hcal_edepthHot5;
  double hcal_activeHotL5;
  double hcal_edepth6;
  double hcal_activeL6;
  double hcal_edepthHot6;
  double hcal_activeHotL6;
  double hcal_edepth7;
  double hcal_activeL7;
  double hcal_edepthHot7;
  double hcal_activeHotL7;

  TBranch *b_Run_No;            //!
  TBranch *b_Event_No;          //!
  TBranch *b_LumiNumber;        //!
  TBranch *b_BXNumber;          //!
  TBranch *b_pt_of_muon;        //!
  TBranch *b_eta_of_muon;       //!
  TBranch *b_phi_of_muon;       //!
  TBranch *b_p_of_muon;         //!
  TBranch *b_ecal_3x3;          //!
  TBranch *b_ecal_detID;        //!
  TBranch *b_hcal_1x1;          //!
  TBranch *b_hcal_detID;        //!
  TBranch *b_hcal_cellHot;      //!
  TBranch *b_activeLength;      //!
  TBranch *b_hcal_edepth1;      //!
  TBranch *b_hcal_edepth2;      //!
  TBranch *b_hcal_edepth3;      //!
  TBranch *b_hcal_edepth4;      //!
  TBranch *b_hcal_activeL1;     //!
  TBranch *b_hcal_activeL2;     //!
  TBranch *b_hcal_activeL3;     //!
  TBranch *b_hcal_activeL4;     //!
  TBranch *b_activeLengthHot;   //!
  TBranch *b_hcal_edepthHot1;   //!
  TBranch *b_hcal_edepthHot2;   //!
  TBranch *b_hcal_edepthHot3;   //!
  TBranch *b_hcal_edepthHot4;   //!
  TBranch *b_hcal_activeHotL1;  //!
  TBranch *b_hcal_activeHotL2;  //!
  TBranch *b_hcal_activeHotL3;  //!
  TBranch *b_hcal_activeHotL4;  //!
  TBranch *b_hcal_edepth5;      //!
  TBranch *b_hcal_activeL5;     //!
  TBranch *b_hcal_edepthHot5;   //!
  TBranch *b_hcal_activeHotL5;  //!
  TBranch *b_hcal_edepth6;      //!
  TBranch *b_hcal_activeL6;     //!
  TBranch *b_hcal_edepthHot6;   //!
  TBranch *b_hcal_activeHotL6;  //!
  TBranch *b_hcal_edepth7;      //!
  TBranch *b_hcal_activeL7;     //!
  TBranch *b_hcal_edepthHot7;   //!
  TBranch *b_hcal_activeHotL7;  //!
  TBranch *b_matchedId;         //!

public:
  HBHEMuonOfflineSimAnalyzer(const char *infile,
                             const char *outfile = "dyll_PU20_25_output.root",
                             const int mode = 0,
                             const int maxDHB = 4,
                             const int maxDHE = 7);
  virtual ~HBHEMuonOfflineSimAnalyzer();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TTree *tree);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);

  std::vector<std::string> firedTriggers;
  void BookHistograms(const char *);
  void WriteHistograms();
  bool LooseMuon();
  bool tightMuon();
  bool SoftMuon();
  void etaPhiHcal(unsigned int detId, int &eta, int &phi, int &depth);
  void etaPhiEcal(unsigned int detId, int &type, int &zside, int &etaX, int &phiY, int &plane, int &strip);
  void calculateP(double pt, double eta, double &pM);
  void close();
  int NDepthBins(int ieta, int iphi);
  int NPhiBins(int ieta);

private:
  static const bool debug_ = false;
  static const int maxDep = 7;
  static const int maxEta = 29;
  static const int maxPhi = 72;
  //3x16x72x2 + 5x4x72x2 + 5x9x36x2
  static const int maxHist = 20000;  //13032;
  int modeLHC_, maxDepthHB_, maxDepthHE_, maxDepth_;
  int nHist, nDepths[maxEta], nDepthsPhi[maxEta], indxEta[maxEta][maxDep][maxPhi];
  TFile *output_file;

  TH1D *h_Pt_Muon[3], *h_Eta_Muon[3], *h_Phi_Muon[3], *h_P_Muon[3];
  TH1D *h_PF_Muon[3], *h_GlobTrack_Chi[3], *h_Global_Muon_Hits[3];
  TH1D *h_MatchedStations[3], *h_Tight_TransImpactparameter[3];
  TH1D *h_Tight_LongitudinalImpactparameter[3], *h_InnerTrackPixelHits[3];
  TH1D *h_TrackerLayer[3], *h_IsolationR04[3], *h_Global_Muon[3];
  TH1D *h_LongImpactParameter[3], *h_LongImpactParameterBin1[3], *h_LongImpactParameterBin2[3];

  TH1D *h_TransImpactParameter[3], *h_TransImpactParameterBin1[3], *h_TransImpactParameterBin2[3];
  TH1D *h_Hot_MuonEnergy_hcal_ClosestCell[3][maxHist], *h_Hot_MuonEnergy_hcal_HotCell[3][maxHist],
      *h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[3][maxHist], *h_HotCell_MuonEnergy_phi[3][maxHist];
  TH2D *h_2D_Bin1[3], *h_2D_Bin2[3];
  TH1D *h_ecal_energy[3], *h_hcal_energy[3], *h_3x3_ecal[3], *h_1x1_hcal[3];
  TH1D *h_MuonHittingEcal[3], *h_HotCell[3], *h_MuonEnergy_hcal[3][maxHist];
  TH1D *h_Hot_MuonEnergy_hcal[3][maxHist];
  TH2D *hcal_ietaVsEnergy[3];
  TProfile *h_EtaX_hcal[3], *h_PhiY_hcal[3], *h_EtaX_ecal[3], *h_PhiY_ecal[3];
  TProfile *h_Eta_ecal[3], *h_Phi_ecal[3];
  TProfile *h_MuonEnergy_eta[3][maxDep], *h_MuonEnergy_phi[3][maxDep], *h_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_Hot_MuonEnergy_eta[3][maxDep], *h_Hot_MuonEnergy_phi[3][maxDep], *h_Hot_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_IsoHot_MuonEnergy_eta[3][maxDep], *h_IsoHot_MuonEnergy_phi[3][maxDep],
      *h_IsoHot_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_IsoWithoutHot_MuonEnergy_eta[3][maxDep], *h_IsoWithoutHot_MuonEnergy_phi[3][maxDep],
      *h_IsoWithoutHot_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_HotWithoutIso_MuonEnergy_eta[3][maxDep], *h_HotWithoutIso_MuonEnergy_phi[3][maxDep],
      *h_HotWithoutIso_MuonEnergy_muon_eta[3][maxDep];
};

HBHEMuonOfflineSimAnalyzer::HBHEMuonOfflineSimAnalyzer(
    const char *infile, const char *outFileName, const int mode, const int maxDHB, const int maxDHE) {
  modeLHC_ = mode;
  maxDepthHB_ = maxDHB;
  maxDepthHE_ = maxDHE;
  maxDepth_ = (maxDepthHB_ > maxDepthHE_) ? maxDepthHB_ : maxDepthHE_;
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  TFile *f = new TFile(infile);
  TDirectory *dir = (TDirectory *)f->Get("HcalHBHEMuonAnalyzer");
  TTree *tree(0);
  dir->GetObject("TREE", tree);
  Init(tree);

  //Now book histograms
  BookHistograms(outFileName);
}

HBHEMuonOfflineSimAnalyzer::~HBHEMuonOfflineSimAnalyzer() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t HBHEMuonOfflineSimAnalyzer::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Int_t HBHEMuonOfflineSimAnalyzer::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t HBHEMuonOfflineSimAnalyzer::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (fChain->GetTreeNumber() != fCurrent) {
    fCurrent = fChain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void HBHEMuonOfflineSimAnalyzer::Init(TTree *tree) {
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
  p_of_muon = 0;
  ecal_3x3 = 0;
  ecal_detID = 0;
  hcal_1x1 = 0;
  hcal_detID = 0;
  hcal_cellHot = 0;
  activeLength = 0;
  hcal_edepth1 = 0;
  hcal_edepth2 = 0;
  hcal_edepth3 = 0;
  hcal_edepth4 = 0;
  hcal_activeL1 = 0;
  hcal_activeL2 = 0;
  hcal_activeL3 = 0;
  hcal_activeL4 = 0;
  activeLengthHot = 0;
  hcal_edepthHot1 = 0;
  hcal_edepthHot2 = 0;
  hcal_edepthHot3 = 0;
  hcal_edepthHot4 = 0;
  hcal_activeHotL1 = 0;
  hcal_activeHotL2 = 0;
  hcal_activeHotL3 = 0;
  hcal_activeHotL4 = 0;
  hcal_edepth5 = 0;
  hcal_activeL5 = 0;
  hcal_edepthHot5 = 0;
  hcal_activeHotL5 = 0;
  hcal_edepth6 = 0;
  hcal_activeL6 = 0;
  hcal_edepthHot6 = 0;
  hcal_activeHotL6 = 0;
  hcal_edepth7 = 0;
  hcal_activeL7 = 0;
  hcal_edepthHot7 = 0;
  hcal_activeHotL7 = 0;
  matchedId = 0;
  if (!tree)
    return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("Run_No", &Run_No, &b_Run_No);
  fChain->SetBranchAddress("Event_No", &Event_No, &b_Event_No);
  fChain->SetBranchAddress("LumiNumber", &LumiNumber, &b_LumiNumber);
  fChain->SetBranchAddress("BXNumber", &BXNumber, &b_BXNumber);
  fChain->SetBranchAddress("pt_of_muon", &pt_of_muon, &b_pt_of_muon);
  fChain->SetBranchAddress("eta_of_muon", &eta_of_muon, &b_eta_of_muon);
  fChain->SetBranchAddress("phi_of_muon", &phi_of_muon, &b_phi_of_muon);
  fChain->SetBranchAddress("p_of_muon", &p_of_muon, &b_p_of_muon);
  fChain->SetBranchAddress("ecal_3x3", &ecal_3x3, &b_ecal_3x3);
  fChain->SetBranchAddress("ecal_detID", &ecal_detID, &b_ecal_detID);
  fChain->SetBranchAddress("hcal_1x1", &hcal_1x1, &b_hcal_1x1);
  fChain->SetBranchAddress("matchedId", &matchedId, &b_matchedId);
  fChain->SetBranchAddress("hcal_detID", &hcal_detID, &b_hcal_detID);
  fChain->SetBranchAddress("hcal_cellHot", &hcal_cellHot, &b_hcal_cellHot);
  fChain->SetBranchAddress("activeLength", &activeLength, &b_activeLength);
  fChain->SetBranchAddress("hcal_edepth1", &hcal_edepth1, &b_hcal_edepth1);
  fChain->SetBranchAddress("hcal_edepth2", &hcal_edepth2, &b_hcal_edepth2);
  fChain->SetBranchAddress("hcal_edepth3", &hcal_edepth3, &b_hcal_edepth3);
  fChain->SetBranchAddress("hcal_edepth4", &hcal_edepth4, &b_hcal_edepth4);
  fChain->SetBranchAddress("hcal_edepth5", &hcal_edepth5, &b_hcal_edepth5);
  fChain->SetBranchAddress("hcal_edepth6", &hcal_edepth6, &b_hcal_edepth6);
  fChain->SetBranchAddress("hcal_edepth7", &hcal_edepth7, &b_hcal_edepth7);
  fChain->SetBranchAddress("hcal_activeL1", &hcal_activeL1, &b_hcal_activeL1);
  fChain->SetBranchAddress("hcal_activeL2", &hcal_activeL2, &b_hcal_activeL2);
  fChain->SetBranchAddress("hcal_activeL3", &hcal_activeL3, &b_hcal_activeL3);
  fChain->SetBranchAddress("hcal_activeL4", &hcal_activeL4, &b_hcal_activeL4);
  fChain->SetBranchAddress("hcal_activeL5", &hcal_activeL5, &b_hcal_activeL5);
  fChain->SetBranchAddress("hcal_activeL6", &hcal_activeL6, &b_hcal_activeL6);
  fChain->SetBranchAddress("hcal_activeL7", &hcal_activeL7, &b_hcal_activeL7);
  fChain->SetBranchAddress("activeLengthHot", &activeLengthHot, &b_activeLengthHot);
  fChain->SetBranchAddress("hcal_edepthHot1", &hcal_edepthHot1, &b_hcal_edepthHot1);
  fChain->SetBranchAddress("hcal_edepthHot2", &hcal_edepthHot2, &b_hcal_edepthHot2);
  fChain->SetBranchAddress("hcal_edepthHot3", &hcal_edepthHot3, &b_hcal_edepthHot3);
  fChain->SetBranchAddress("hcal_edepthHot4", &hcal_edepthHot4, &b_hcal_edepthHot4);
  fChain->SetBranchAddress("hcal_edepthHot5", &hcal_edepthHot5, &b_hcal_edepthHot5);
  fChain->SetBranchAddress("hcal_edepthHot6", &hcal_edepthHot6, &b_hcal_edepthHot6);
  fChain->SetBranchAddress("hcal_edepthHot7", &hcal_edepthHot7, &b_hcal_edepthHot7);
  fChain->SetBranchAddress("hcal_activeHotL1", &hcal_activeHotL1, &b_hcal_activeHotL1);
  fChain->SetBranchAddress("hcal_activeHotL2", &hcal_activeHotL2, &b_hcal_activeHotL2);
  fChain->SetBranchAddress("hcal_activeHotL3", &hcal_activeHotL3, &b_hcal_activeHotL3);
  fChain->SetBranchAddress("hcal_activeHotL4", &hcal_activeHotL4, &b_hcal_activeHotL4);
  fChain->SetBranchAddress("hcal_activeHotL5", &hcal_activeHotL5, &b_hcal_activeHotL5);
  fChain->SetBranchAddress("hcal_activeHotL6", &hcal_activeHotL6, &b_hcal_activeHotL6);
  fChain->SetBranchAddress("hcal_activeHotL7", &hcal_activeHotL7, &b_hcal_activeHotL7);
  Notify();
}

void HBHEMuonOfflineSimAnalyzer::Loop() {
  //declarations
  if (fChain == 0)
    return;

  Long64_t nentries = fChain->GetEntriesFast();

  if (debug_)
    std::cout << "nevent = " << nentries << std::endl;

  Long64_t nbytes = 0, nb = 0;

  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;

    if (debug_) {
      std::cout << "ecal_det_id " << ecal_detID << std::endl;
      std::cout << "hcal_det_id " << std::hex << hcal_detID << std::dec;
    }
    int etaHcal, phiHcal, depthHcal;
    etaPhiHcal(hcal_detID, etaHcal, phiHcal, depthHcal);

    int eta = (etaHcal > 0) ? (etaHcal - 1) : -(1 + etaHcal);
    int nDepth = NDepthBins(eta + 1, phiHcal);
    int nPhi = NPhiBins(eta + 1);

    double phiYHcal = (phiHcal - 0.5);
    if (debug_)
      std::cout << "phiHcal" << phiHcal << " phiYHcal" << phiYHcal << std::endl;

    for (int cut = 0; cut < 3; ++cut) {
      bool select(false);
      if (cut == 0)
        select = tightMuon();
      else if (cut == 1)
        select = SoftMuon();
      else
        select = LooseMuon();

      if (select) {
        //	  h_P_Muon[cut]->Fill(p_of_muon);
        h_P_Muon[cut]->Fill(p_of_muon);
        h_Pt_Muon[cut]->Fill(pt_of_muon);
        h_Eta_Muon[cut]->Fill(eta_of_muon);

        double energyFill;
        for (int dep = 0; dep < nDepth; ++dep) {
          if (debug_) {
            std::cout << "why on 15/2 only" << std::endl;
            std::cout << "dep:" << dep << std::endl;
          }
          int PHI = (nPhi > 36) ? (phiHcal - 1) : (phiHcal - 1) / 2;
          double en1(-9999), en2(-9999);
          if (dep == 0) {
            en1 = hcal_edepth1;
            en2 = hcal_edepthHot1;
            energyFill = (hcal_activeHotL1 > 0) ? hcal_activeHotL1 : 999;
          } else if (dep == 1) {
            en1 = hcal_edepth2;
            en2 = hcal_edepthHot2;
            energyFill = (hcal_activeHotL2 > 0) ? hcal_activeHotL2 : 999;
            if (debug_)
              std::cout << "problem here.. lets see if it got printed\n";
          } else if (dep == 2) {
            en1 = hcal_edepth3;
            en2 = hcal_edepthHot3;
            energyFill = (hcal_activeHotL3 > 0) ? hcal_activeHotL3 : 999;
          } else if (dep == 3) {
            en1 = hcal_edepth4;
            en2 = hcal_edepthHot4;
            if (debug_)
              std::cout << "Hello in 4" << std::endl;
            energyFill = (hcal_activeHotL4 > 0) ? hcal_activeHotL4 : 999;
          } else if (dep == 4) {
            en1 = hcal_edepth5;
            en2 = hcal_edepthHot5;
            energyFill = (hcal_activeHotL5 > 0) ? hcal_activeHotL5 : 999;
          } else if (dep == 5) {
            if (debug_)
              std::cout << "Energy in depth 6 " << maxDepth_ << ":" << hcal_edepth6 << ":" << hcal_edepthHot6
                        << std::endl;
            en1 = (maxDepth_ > 5) ? hcal_edepth6 : 0;
            en2 = (maxDepth_ > 5) ? hcal_edepthHot6 : 0;
            energyFill = (hcal_activeHotL6 > 0) ? hcal_activeHotL6 : 999;
          } else if (dep == 6) {
            if (debug_)
              std::cout << "Energy in depth 7 " << maxDepth_ << ":" << hcal_edepth7 << ":" << hcal_edepthHot7
                        << std::endl;
            en1 = (maxDepth_ > 6) ? hcal_edepth7 : 0;
            en2 = (maxDepth_ > 6) ? hcal_edepthHot7 : 0;
            energyFill = (hcal_activeHotL7 > 0) ? hcal_activeHotL7 : 999;
          }

          if (debug_) {
            std::cout << " Debug2" << std::endl;
            std::cout << "ok1" << en1 << std::endl;
            std::cout << "ok2" << en2 << std::endl;
          }
          bool ok1 = (en1 > -9999);
          bool ok2 = (en2 > -9999);

          if (debug_)
            std::cout << "Before Index" << std::endl;

          int ind = (etaHcal > 0) ? indxEta[eta][dep][PHI] : 1 + indxEta[eta][dep][PHI];

          if (debug_) {
            std::cout << "ieta " << eta << "depth " << dep << "indxEta[eta][dep]:" << indxEta[eta][dep][PHI]
                      << std::endl;
            std::cout << "index showing eta,depth:" << ind << std::endl;
            std::cout << "etaHcal: " << etaHcal << " eta " << eta << " dep " << dep << " indx " << ind << std::endl;
          }
          if (!(matchedId))
            continue;
          if (ok1) {
            if (debug_)
              std::cout << "enter ok1" << std::endl;
            if (hcal_cellHot == 1) {
              if (en2 > 0) {
                h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[cut][ind]->Fill(en2 / energyFill);
              }
              if (debug_)
                std::cout << "enter hot cell" << std::endl;
            }
          }

          if (ok2) {
            if (debug_)
              std::cout << "enter ok2" << std::endl;
            if (hcal_cellHot != 1) {
            }
          }

          if (debug_)
            std::cout << "ETA \t" << eta << "DEPTH \t" << dep << std::endl;
        }
      }
    }
  }
  close();
}

Bool_t HBHEMuonOfflineSimAnalyzer::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void HBHEMuonOfflineSimAnalyzer::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

void HBHEMuonOfflineSimAnalyzer::BookHistograms(const char *fname) {
  output_file = TFile::Open(fname, "RECREATE");
  //output_file->cd();
  std::string type[] = {"tight", "soft", "loose"};
  char name[128], title[500];

  std::cout << "BookHistograms" << std::endl;

  nHist = 0;
  for (int eta = 0; eta < 29; ++eta) {
    int nDepth = NDepthBins(eta + 1, -1);
    int nPhi = NPhiBins(eta + 1);
    for (int depth = 0; depth < nDepth; depth++) {
      for (int PHI = 0; PHI < nPhi; ++PHI) {
        indxEta[eta][depth][PHI] = nHist;
        nHist += 2;
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    sprintf(name, "h_Pt_Muon_%s", type[i].c_str());
    sprintf(title, "p_{T} of %s muons (GeV)", type[i].c_str());
    h_Pt_Muon[i] = new TH1D(name, title, 100, 0, 200);

    sprintf(name, "h_Eta_Muon_%s", type[i].c_str());
    sprintf(title, "#eta of %s muons", type[i].c_str());
    h_Eta_Muon[i] = new TH1D(name, title, 50, -2.5, 2.5);

    sprintf(name, "h_Phi_Muon_%s", type[i].c_str());
    sprintf(title, "#phi of %s muons", type[i].c_str());
    h_Phi_Muon[i] = new TH1D(name, title, 100, -3.1415926, 3.1415926);

    sprintf(name, "h_P_Muon_%s", type[i].c_str());
    sprintf(title, "p of %s muons (GeV)", type[i].c_str());
    h_P_Muon[i] = new TH1D(name, title, 100, 0, 200);

    sprintf(name, "h_PF_Muon_%s", type[i].c_str());
    sprintf(title, "PF %s muons (GeV)", type[i].c_str());
    h_PF_Muon[i] = new TH1D(name, title, 2, 0, 2);

    sprintf(name, "h_Global_Muon_Chi2_%s", type[i].c_str());
    sprintf(title, "Chi2 Global %s muons (GeV)", type[i].c_str());
    h_GlobTrack_Chi[i] = new TH1D(name, title, 15, 0, 15);

    sprintf(name, "h_Global_Muon_Hits_%s", type[i].c_str());
    sprintf(title, "Global Hits %s muons (GeV)", type[i].c_str());
    h_Global_Muon_Hits[i] = new TH1D(name, title, 10, 0, 10);

    sprintf(name, "h_Matched_Stations_%s", type[i].c_str());
    sprintf(title, "Matched Stations %s muons (GeV)", type[i].c_str());
    h_MatchedStations[i] = new TH1D(name, title, 10, 0, 10);

    sprintf(name, "h_Transverse_ImpactParameter_%s", type[i].c_str());
    sprintf(title, "Transverse_ImpactParameter of %s muons (GeV)", type[i].c_str());
    h_Tight_TransImpactparameter[i] = new TH1D(name, title, 50, 0, 10);

    sprintf(name, "h_Longitudinal_ImpactParameter_%s", type[i].c_str());
    sprintf(title, "Longitudinal_ImpactParameter of %s muons (GeV)", type[i].c_str());
    h_Tight_LongitudinalImpactparameter[i] = new TH1D(name, title, 20, 0, 10);

    sprintf(name, "h_InnerTrack_PixelHits_%s", type[i].c_str());
    sprintf(title, "InnerTrack_PixelHits of %s muons (GeV)", type[i].c_str());
    h_InnerTrackPixelHits[i] = new TH1D(name, title, 20, 0, 20);

    sprintf(name, "h_TrackLayers_%s", type[i].c_str());
    sprintf(title, "No. of Tracker Layers of %s muons (GeV)", type[i].c_str());
    h_TrackerLayer[i] = new TH1D(name, title, 20, 0, 20);
    ;

    sprintf(name, "h_IsolationR04_%s", type[i].c_str());
    sprintf(title, "IsolationR04 %s muons (GeV)", type[i].c_str());
    h_IsolationR04[i] = new TH1D(name, title, 45, 0, 5);
    ;

    sprintf(name, "h_Global_Muon_%s", type[i].c_str());
    sprintf(title, "Global %s muons (GeV)", type[i].c_str());
    h_Global_Muon[i] = new TH1D(name, title, 2, 0, 2);

    sprintf(name, "h_TransImpactParameter_%s", type[i].c_str());
    sprintf(title, "TransImpactParameter of %s muons (GeV)", type[i].c_str());
    h_TransImpactParameter[i] = new TH1D(name, title, 100, 0, 0.5);

    sprintf(name, "h_TransImpactParameterBin1_%s", type[i].c_str());
    sprintf(title, "TransImpactParameter of %s muons (GeV) in -1.5 <= #phi <= 0.5", type[i].c_str());
    h_TransImpactParameterBin1[i] = new TH1D(name, title, 100, 0, 0.5);

    sprintf(name, "h_TransImpactParameterBin2_%s", type[i].c_str());
    sprintf(title, "TransImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
    h_TransImpactParameterBin2[i] = new TH1D(name, title, 100, 0, 0.5);
    //
    sprintf(name, "h_LongImpactParameter_%s", type[i].c_str());
    sprintf(title, "LongImpactParameter of %s muons (GeV)", type[i].c_str());
    h_LongImpactParameter[i] = new TH1D(name, title, 100, 0, 30);

    sprintf(name, "h_LongImpactParameterBin1_%s", type[i].c_str());
    sprintf(title, "LongImpactParameter of %s muons (GeV) in -1.5 <= #phi <= 0.5", type[i].c_str());
    h_LongImpactParameterBin1[i] = new TH1D(name, title, 100, 0, 30);

    sprintf(name, "h_LongImpactParameterBin2_%s", type[i].c_str());
    sprintf(title, "LongImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
    h_LongImpactParameterBin2[i] = new TH1D(name, title, 100, 0, 30);

    sprintf(name, "h_2D_Bin1_%s", type[i].c_str());
    sprintf(title, "Trans/Long ImpactParameter of %s muons (GeV) in -1.5 <= #phi< 0.5 ", type[i].c_str());
    h_2D_Bin1[i] = new TH2D(name, title, 100, 0, 0.5, 100, 0, 30);

    sprintf(name, "h_2D_Bin2_%s", type[i].c_str());
    sprintf(title, "Trans/Long ImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
    h_2D_Bin2[i] = new TH2D(name, title, 100, 0, 0.5, 100, 0, 30);

    sprintf(name, "h_ecal_energy_%s", type[i].c_str());
    sprintf(title, "ECAL energy for %s muons", type[i].c_str());
    h_ecal_energy[i] = new TH1D(name, title, 1000, -10.0, 90.0);

    sprintf(name, "h_hcal_energy_%s", type[i].c_str());
    sprintf(title, "HCAL energy for %s muons", type[i].c_str());
    h_hcal_energy[i] = new TH1D(name, title, 500, -10.0, 90.0);

    sprintf(name, "h_3x3_ecal_%s", type[i].c_str());
    sprintf(title, "ECAL energy in 3x3 for %s muons", type[i].c_str());
    h_3x3_ecal[i] = new TH1D(name, title, 1000, -10.0, 90.0);

    sprintf(name, "h_1x1_hcal_%s", type[i].c_str());
    sprintf(title, "HCAL energy in 1x1 for %s muons", type[i].c_str());
    h_1x1_hcal[i] = new TH1D(name, title, 500, -10.0, 90.0);

    sprintf(name, "h_EtaX_hcal_%s", type[i].c_str());
    sprintf(title, "HCAL energy as a function of i#eta for %s muons", type[i].c_str());
    h_EtaX_hcal[i] = new TProfile(name, title, 60, -30.0, 30.0);

    sprintf(name, "h_PhiY_hcal_%s", type[i].c_str());
    sprintf(title, "HCAL energy as a function of i#phi for %s muons", type[i].c_str());
    h_PhiY_hcal[i] = new TProfile(name, title, 72, 0, 72);

    sprintf(name, "h_EtaX_ecal_%s", type[i].c_str());
    sprintf(title, "EB energy as a function of i#eta for %s muons", type[i].c_str());
    h_EtaX_ecal[i] = new TProfile(name, title, 170, -85.0, 85.0);

    sprintf(name, "h_PhiY_ecal_%s", type[i].c_str());
    sprintf(title, "EB energy as a function of i#phi for %s muons", type[i].c_str());
    h_PhiY_ecal[i] = new TProfile(name, title, 360, 0, 360);

    sprintf(name, "h_Eta_ecal_%s", type[i].c_str());
    sprintf(title, "ECAL energy as a function of #eta for %s muons", type[i].c_str());
    h_Eta_ecal[i] = new TProfile(name, title, 100, -2.5, 2.5);

    sprintf(name, "h_Phi_ecal_%s", type[i].c_str());
    sprintf(title, "ECAL energy as a function of #phi for %s muons", type[i].c_str());
    h_Phi_ecal[i] = new TProfile(name, title, 100, -3.1415926, 3.1415926);

    sprintf(name, "h_MuonHittingEcal_%s", type[i].c_str());
    sprintf(title, "%s muons hitting ECAL", type[i].c_str());
    h_MuonHittingEcal[i] = new TH1D(name, title, 100, 0, 5.0);

    sprintf(name, "h_HotCell_%s", type[i].c_str());
    sprintf(title, "Hot cell for %s muons", type[i].c_str());
    h_HotCell[i] = new TH1D(name, title, 100, 0, 2);

    std::cout << "problem here" << std::endl;
    for (int eta = 0; eta < 29; ++eta) {
      int nDepth = NDepthBins(eta + 1, -1);
      int nPhi = NPhiBins(eta + 1);
      for (int depth = 0; depth < nDepth; ++depth) {
        for (int PHI = 0; PHI < nPhi; ++PHI) {
          int PHI0 = (nPhi == 72) ? PHI + 1 : 2 * PHI + 1;
          int ih = indxEta[eta][depth][PHI];
          std::cout << "eta:" << eta << " depth:" << depth << " PHI:" << PHI << ":" << PHI0 << " ih:" << ih
                    << std::endl;

          sprintf(name,
                  "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell_ByActiveLength",
                  (eta + 1),
                  (depth + 1),
                  PHI0,
                  type[i].c_str());
          sprintf(title,
                  "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi = %d) for extrapolated %s muons (Hot Cell) "
                  "divided by Active Length",
                  (eta + 1),
                  (depth + 1),
                  PHI0,
                  type[i].c_str());
          h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih] = new TH1D(name, title, 4000, 0.0, 1.0);
          h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih]->Sumw2();

          ih++;
          sprintf(name,
                  "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell_ByActiveLength",
                  -(eta + 1),
                  (depth + 1),
                  PHI0,
                  type[i].c_str());
          sprintf(title,
                  "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi=%d) for extrapolated %s muons (Hot Cell) "
                  "divided by Active Length",
                  -(eta + 1),
                  (depth + 1),
                  PHI0,
                  type[i].c_str());
          h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih] = new TH1D(name, title, 4000, 0.0, 1.0);
          h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih]->Sumw2();
        }
      }
      //output_file->cd();
    }
  }
  //output_file->cd();
}

bool HBHEMuonOfflineSimAnalyzer::LooseMuon() {
  if (pt_of_muon > 20.) {
    if (fabs(eta_of_muon) <= 5.0) {
      return true;
    }
  }

  return false;
}

bool HBHEMuonOfflineSimAnalyzer::SoftMuon() {
  if (pt_of_muon > 20.) {
    if (fabs(eta_of_muon) <= 5.0) {
      return true;
    }
  }

  return false;
}

bool HBHEMuonOfflineSimAnalyzer::tightMuon() {
  if (pt_of_muon > 20.) {
    if (fabs(eta_of_muon) <= 5.0) {
      return true;
    }
  }

  return false;
}

void HBHEMuonOfflineSimAnalyzer::etaPhiHcal(unsigned int detId, int &eta, int &phi, int &depth) {
  int zside, etaAbs;
  if ((detId & 0x1000000) == 0) {
    zside = (detId & 0x2000) ? (1) : (-1);
    etaAbs = (detId >> 7) & 0x3F;
    phi = detId & 0x7F;
    depth = (detId >> 14) & 0x1F;
  } else {
    zside = (detId & 0x80000) ? (1) : (-1);
    etaAbs = (detId >> 10) & 0x1FF;
    phi = detId & 0x3FF;
    depth = (detId >> 20) & 0xF;
  }
  eta = etaAbs * zside;
}

void HBHEMuonOfflineSimAnalyzer::etaPhiEcal(
    unsigned int detId, int &type, int &zside, int &etaX, int &phiY, int &plane, int &strip) {
  type = ((detId >> 25) & 0x7);
  plane = strip = 0;
  if (type == 1) {
    //Ecal Barrel
    zside = (detId & 0x10000) ? (1) : (-1);
    etaX = (detId >> 9) & 0x7F;
    phiY = detId & 0x1FF;
  } else if (type == 2) {
    zside = (detId & 0x4000) ? (1) : (-1);
    etaX = (detId >> 7) & 0x7F;
    phiY = (detId & 0x7F);
  } else if (type == 3) {
    zside = (detId & 0x80000) ? (1) : (-1);
    etaX = (detId >> 6) & 0x3F;
    /** get the sensor iy */
    phiY = (detId >> 12) & 0x3F;
    /** get the strip */
    plane = ((detId >> 18) & 0x1) + 1;
    strip = detId & 0x3F;
  } else {
    zside = etaX = phiY = 0;
  }
}

void HBHEMuonOfflineSimAnalyzer::calculateP(double pt, double eta, double &pM) {
  pM = (pt * cos(2 * (1 / atan(exp(eta)))));
}

void HBHEMuonOfflineSimAnalyzer::close() {
  output_file->cd();
  std::cout << "file yet to be Written" << std::endl;
  WriteHistograms();
  //	output_file->Write();
  std::cout << "file Written" << std::endl;
  output_file->Close();
  std::cout << "now doing return" << std::endl;
}

void HBHEMuonOfflineSimAnalyzer::WriteHistograms() {
  std::string type[] = {"tight", "soft", "loose"};
  char name[128];

  std::cout << "WriteHistograms" << std::endl;
  nHist = 0;
  for (int eta = 0; eta < 29; ++eta) {
    int nDepth = NDepthBins(eta + 1, -1);
    int nPhi = NPhiBins(eta + 1);
    if (debug_)
      std::cout << "Eta:" << eta << " nDepths " << nDepth << " nPhis " << nPhi << std::endl;
    for (int depth = 0; depth < nDepth; ++depth) {
      if (debug_)
        std::cout << "Eta:" << eta << "Depth:" << depth << std::endl;
      for (int PHI = 0; PHI < nPhi; ++PHI) {
        indxEta[eta][depth][PHI] = nHist;
        nHist += 2;
      }
    }
  }

  TDirectory *d_output_file[3][29];
  for (int i = 0; i < 3; ++i) {
    h_Pt_Muon[i]->Write();
    h_Eta_Muon[i]->Write();
    h_Phi_Muon[i]->Write();
    h_P_Muon[i]->Write();
    h_PF_Muon[i]->Write();

    h_GlobTrack_Chi[i]->Write();
    h_Global_Muon_Hits[i]->Write();
    h_MatchedStations[i]->Write();

    h_Tight_TransImpactparameter[i]->Write();
    h_Tight_LongitudinalImpactparameter[i]->Write();

    h_InnerTrackPixelHits[i]->Write();
    h_TrackerLayer[i]->Write();
    h_IsolationR04[i]->Write();

    h_Global_Muon[i]->Write();
    h_TransImpactParameter[i]->Write();
    ;
    h_TransImpactParameterBin1[i]->Write();
    h_TransImpactParameterBin2[i]->Write();
    //
    h_LongImpactParameter[i]->Write();
    h_LongImpactParameterBin1[i]->Write();
    h_LongImpactParameterBin2[i]->Write();

    h_ecal_energy[i]->Write();
    h_hcal_energy[i]->Write();
    ;
    h_3x3_ecal[i]->Write();
    h_1x1_hcal[i]->Write();
    ;

    h_EtaX_hcal[i]->Write();
    h_PhiY_hcal[i]->Write();
    ;

    h_EtaX_ecal[i]->Write();
    ;
    h_PhiY_ecal[i]->Write();
    ;
    h_Eta_ecal[i]->Write();
    ;
    h_Phi_ecal[i]->Write();
    ;
    h_MuonHittingEcal[i]->Write();
    ;
    h_HotCell[i]->Write();
    ;

    output_file->cd();
    for (int eta = 0; eta < 29; ++eta) {
      int nDepth = NDepthBins(eta + 1, -1);
      int nPhi = NPhiBins(eta + 1);
      sprintf(name, "Dir_muon_type_%s_ieta%d", type[i].c_str(), eta + 1);
      d_output_file[i][eta] = output_file->mkdir(name);
      d_output_file[i][eta]->cd();
      for (int depth = 0; depth < nDepth; ++depth) {
        for (int PHI = 0; PHI < nPhi; ++PHI) {
          int ih = indxEta[eta][depth][PHI];
          h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih]->Write();
          ih++;
          h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih]->Write();
        }
      }
      output_file->cd();
    }
  }
  output_file->cd();
}

int HBHEMuonOfflineSimAnalyzer::NDepthBins(int eta, int phi) {
  // Run 1 scenario
  int nDepthR1[29] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2};
  // Run 2 scenario from 2018
  int nDepthR2[29] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 3, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 3};
  // Run 3 scenario
  int nDepthR3[29] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 3};
  // Run 4 scenario
  int nDepthR4[29] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  // for a test scenario with multi depth segmentation considered during Run 1
  //    int  nDepth[29]={3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5};
  // modeLHC_ = 0 --> nbin defined maxDepthHB/HE
  //          = 1 -->      corresponds to Run 1 (valid till 2016)
  //          = 2 -->      corresponds to Run 2 (2018 geometry)
  //          = 3 -->      corresponds to Run 3 (post LS2)
  //          = 4 -->      corresponds to 2017 (Plan 1)
  //          = 5 -->      corresponds to Run 4 (post LS3)
  int nbin(0);
  if (modeLHC_ == 0) {
    if (eta <= 15) {
      nbin = maxDepthHB_;
    } else if (eta == 16) {
      nbin = 4;
    } else {
      nbin = maxDepthHE_;
    }
  } else if (modeLHC_ == 1) {
    nbin = nDepthR1[eta - 1];
  } else if (modeLHC_ == 2) {
    nbin = nDepthR2[eta - 1];
  } else if (modeLHC_ == 3) {
    nbin = nDepthR3[eta - 1];
  } else if (modeLHC_ == 4) {
    if (phi > 0) {
      if (eta >= 16 && phi >= 63 && phi <= 66) {
        nbin = nDepthR2[eta - 1];
      } else {
        nbin = nDepthR1[eta - 1];
      }
    } else {
      if (eta >= 16) {
        nbin = (nDepthR2[eta - 1] > nDepthR1[eta - 1]) ? nDepthR2[eta - 1] : nDepthR1[eta - 1];
      } else {
        nbin = nDepthR1[eta - 1];
      }
    }
  } else {
    if (eta > 0 && eta < 30) {
      nbin = nDepthR4[eta - 1];
    } else {
      nbin = nDepthR4[28];
    }
  }
  return nbin;
}

int HBHEMuonOfflineSimAnalyzer::NPhiBins(int eta) {
  int nphi = (eta <= 20) ? 72 : 36;
  return nphi;
}
