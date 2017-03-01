//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sat Nov 17 12:34:52 2012 by ROOT version 5.30/04
// from TTree TREE/TREE
// found on file: Validation.root
//////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

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

class HBHEMuonOfflineAnalyzer {

public :
  TTree                *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t                 fCurrent; //!current Tree number in a TChain
  
  // Declaration of leaf types

  UInt_t                Event_No;
  UInt_t                Run_No;
  UInt_t                LumiNumber;
  UInt_t                BXNumber;
  std::vector<double>  *pt_of_muon;
  std::vector<double>  *eta_of_muon;
  std::vector<double>  *phi_of_muon;
  std::vector<double>  *energy_of_muon;
  std::vector<double>  *p_of_muon;
  std::vector<bool>    *PF_Muon;
  std::vector<bool>    *Global_Muon;
  std::vector<bool>    *Tracker_muon;
  std::vector<double>  *hcal_3into3;
  std::vector<double>  *hcal_1x1;
  std::vector<unsigned int> *hcal_detID;
  std::vector<double>  *hcal_edepth1;
  std::vector<double>  *hcal_edepth2;
  std::vector<double>  *hcal_edepth3;
  std::vector<double>  *hcal_edepth4;
  std::vector<double>  *hcal_edepthHot1;
  std::vector<double>  *hcal_edepthHot2;
  std::vector<double>  *hcal_edepthHot3;
  std::vector<double>  *hcal_edepthHot4;
  std::vector<double>  *TrackerLayer;
  std::vector<double>  *matchedId;
  std::vector<bool>    *innerTrack;
  std::vector<double>  *innerTrackpt;
  std::vector<double>  *innerTracketa;
  std::vector<double>  *innerTrackphi;
  std::vector<double>  *MatchedStat;
  std::vector<double>  *GlobalTrckPt;
  std::vector<double>  *GlobalTrckEta;
  std::vector<double>  *GlobalTrckPhi;
  std::vector<double>  *NumPixelLayers;
  std::vector<double>  *chiTracker;
  std::vector<double>  *DxyTracker;
  std::vector<double>  *DzTracker;
  std::vector<bool>    *OuterTrack;
  std::vector<double>  *OuterTrackPt;
  std::vector<double>  *OuterTrackEta;
  std::vector<double>  *OuterTrackPhi;
  std::vector<double>  *OuterTrackHits;
  std::vector<double>  *OuterTrackRHits;
  std::vector<double>  *OuterTrackChi;
  std::vector<bool>    *GlobalTrack;
  std::vector<double>  *GlobTrack_Chi;
  std::vector<double>  *Global_Muon_Hits;
  std::vector<double>  *MatchedStations;
  std::vector<double>  *Global_Track_Pt;
  std::vector<double>  *Global_Track_Eta;
  std::vector<double>  *Global_Track_Phi;
  std::vector<double>  *Tight_LongitudinalImpactparameter;
  std::vector<double>  *Tight_TransImpactparameter;
  std::vector<double>  *InnerTrackPixelHits;
  std::vector<double>  *IsolationR04;
  std::vector<double>  *IsolationR03;
  std::vector<double>  *hcal_cellHot;
  std::vector<double>  *ecal_3into3;
  std::vector<double>  *ecal_3x3;
  std::vector<unsigned int> *ecal_detID;
  std::vector<unsigned int> *ehcal_detID;
  std::vector<double>  *tracker_3into3;
  std::vector<double>  *activeLength;
  std::vector<int>     *hltresults;
  std::vector<string>  *all_triggers;


  TBranch              *b_Event_No;   //!
  TBranch              *b_Run_No;   //!
  TBranch              *b_LumiNumber;   //!
  TBranch              *b_BXNumber;   //!
  TBranch              *b_pt_of_muon;   //!
  TBranch              *b_eta_of_muon;   //!
  TBranch              *b_phi_of_muon;   //!
  TBranch              *b_energy_of_muon;   //!
  TBranch              *b_p_of_muon;   //!
  TBranch              *b_PF_Muon;   //!
  TBranch              *b_Global_Muon;   //!
  TBranch              *b_Tracker_muon;   //!
  TBranch              *b_hcal_3into3;   //!
  TBranch              *b_hcal_1x1;   //!
  TBranch              *b_hcal_detID;   //!
  TBranch              *b_hcal_edepth1;   //!
  TBranch              *b_hcal_edepth2;   //!
  TBranch              *b_hcal_edepth3;   //!
  TBranch              *b_hcal_edepth4;   //!
  TBranch              *b_hcal_edepthHot1;   //!
  TBranch              *b_hcal_edepthHot2;   //!
  TBranch              *b_hcal_edepthHot3;   //!
  TBranch              *b_hcal_edepthHot4;   //!
  TBranch              *b_TrackerLayer;   //!
  TBranch              *b_matchedId;   //!
  TBranch              *b_innerTrack;   //!
  TBranch              *b_innerTrackpt;   //!
  TBranch              *b_innerTracketa;   //!
  TBranch              *b_innerTrackphi;   //!
  TBranch              *b_MatchedStat;   //!
  TBranch              *b_GlobalTrckPt;   //!
  TBranch              *b_GlobalTrckEta;   //!
  TBranch              *b_GlobalTrckPhi;   //!
  TBranch              *b_NumPixelLayers;   //!
  TBranch              *b_chiTracker;   //!
  TBranch              *b_DxyTracker;   //!
  TBranch              *b_DzTracker;   //!
  TBranch              *b_OuterTrack;   //!
  TBranch              *b_OuterTrackPt;   //!
  TBranch              *b_OuterTrackEta;   //!
  TBranch              *b_OuterTrackPhi;   //!
  TBranch              *b_OuterTrackHits;   //!
  TBranch              *b_OuterTrackRHits;   //!
  TBranch              *b_OuterTrackChi;   //!
  TBranch              *b_GlobalTrack;   //!
  TBranch              *b_GlobTrack_Chi;   //!
  TBranch              *b_Global_Muon_Hits;   //!
  TBranch              *b_MatchedStations;   //!
  TBranch              *b_Global_Track_Pt;   //!
  TBranch              *b_Global_Track_Eta;   //!
  TBranch              *b_Global_Track_Phi;   //!
  TBranch              *b_Tight_LongitudinalImpactparameter;   //!
  TBranch              *b_Tight_TransImpactparameter;   //!
  TBranch              *b_InnerTrackPixelHits;   //!
  TBranch              *b_IsolationR04;   //!
  TBranch              *b_IsolationR03;   //!
  TBranch              *b_hcal_cellHot;   //!
  TBranch              *b_ecal_3into3;   //!
  TBranch              *b_ecal_3x3;   //!
  TBranch              *b_ecal_detID;   //!
  TBranch              *b_ehcal_detID;   //!
  TBranch              *b_tracker_3into3;   //!
  TBranch              *b_activeLength;   //!
  TBranch              *b_hltresults;   //!
  TBranch              *b_all_triggers;   //!


  HBHEMuonOfflineAnalyzer(TTree *tree=0, const char *outfile="dyll_PU20_25_output_10.root", const int mode=0, const int maxDHB=3, const int maxDHE=3);
  // mode of LHC is kept 1 for 2017 scenario as no change in depth segmentation
  // mode of LHC is 0 for 2019
  virtual ~HBHEMuonOfflineAnalyzer();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  
  std::vector<std::string> firedTriggers;
  void BookHistograms(const char* );
  bool LooseMuon(unsigned int ml);
  bool tightMuon(unsigned int ml);
  bool SoftMuon(unsigned int ml);
  void etaPhiHcal(unsigned int detId, int &eta, int &phi, int &depth);
  void etaPhiEcal(unsigned int detId, int& type, int& zside,
		  int& etaX, int& phiY, int& plane, int& strip);
  void calculateP(double pt ,double eta , double& pM);
  void close();
private:
  static const bool debug_=false;
  static const int maxDep=3;
  static const int maxEta=29;
  static const int maxPhi=72;
  //3x16x72x2 + 5x4x72x2 + 5x9x36x2
  static const int maxHist=13032;
  int    modeLHC, maxDepthHB_, maxDepthHE_, maxDepth_;
  int    nHist, nDepths[maxEta], nDepthsPhi[maxEta],indxEta[maxEta][maxDep][maxPhi];
  TFile *output_file;

  TH1D  *h_Pt_Muon[3], *h_Eta_Muon[3], *h_Phi_Muon[3], *h_P_Muon[3];
  TH1D  *h_PF_Muon[3], *h_GlobTrack_Chi[3], *h_Global_Muon_Hits[3];
  TH1D  *h_MatchedStations[3], *h_Tight_TransImpactparameter[3];
  TH1D  *h_Tight_LongitudinalImpactparameter[3], *h_InnerTrackPixelHits[3];
  TH1D  *h_TrackerLayer[3], *h_IsolationR04[3] , *h_Global_Muon[3];
  TH1D *h_LongImpactParameter[3], *h_LongImpactParameterBin1[3], *h_LongImpactParameterBin2[3];
  
  TH1D *h_TransImpactParameter[3], *h_TransImpactParameterBin1[3], *h_TransImpactParameterBin2[3];
  TH1D *h_Hot_MuonEnergy_hcal_ClosestCell[3][maxHist] , *h_Hot_MuonEnergy_hcal_HotCell[3][maxHist] , *h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[3][maxHist], *h_HotCell_MuonEnergy_phi[3][maxHist]; 
  TH2D  *h_2D_Bin1[3], *h_2D_Bin2[3];
  TH1D  *h_ecal_energy[3], *h_hcal_energy[3], *h_3x3_ecal[3], *h_1x1_hcal[3];
  TH1D  *h_MuonHittingEcal[3], *h_HotCell[3], *h_MuonEnergy_hcal[3][maxHist];
  TH1D  *h_Hot_MuonEnergy_hcal[3][maxHist];
  TH2D  *hcal_ietaVsEnergy[3];
  TProfile *h_EtaX_hcal[3], *h_PhiY_hcal[3], *h_EtaX_ecal[3], *h_PhiY_ecal[3];
  TProfile *h_Eta_ecal[3], *h_Phi_ecal[3];
  TProfile *h_MuonEnergy_eta[3][maxDep], *h_MuonEnergy_phi[3][maxDep], *h_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_Hot_MuonEnergy_eta[3][maxDep], *h_Hot_MuonEnergy_phi[3][maxDep],  *h_Hot_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_IsoHot_MuonEnergy_eta[3][maxDep], *h_IsoHot_MuonEnergy_phi[3][maxDep], *h_IsoHot_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_IsoWithoutHot_MuonEnergy_eta[3][maxDep], *h_IsoWithoutHot_MuonEnergy_phi[3][maxDep], *h_IsoWithoutHot_MuonEnergy_muon_eta[3][maxDep];
  TProfile *h_HotWithoutIso_MuonEnergy_eta[3][maxDep], *h_HotWithoutIso_MuonEnergy_phi[3][maxDep], *h_HotWithoutIso_MuonEnergy_muon_eta[3][maxDep];
};

HBHEMuonOfflineAnalyzer::HBHEMuonOfflineAnalyzer(TTree *tree, const char* outFileName, 
				 const int mode, const int maxDHB,
				 const int maxDHE) : modeLHC(mode),
						     maxDepthHB_(maxDHB),
						     maxDepthHE_(maxDHE) {
  maxDepth_ = (maxDepthHB_ > maxDepthHE_) ? maxDepthHB_ : maxDepthHE_;
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  // std::cout<<"maxDepth_"<<maxDepth_<<std::endl;
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("file:/uscmst1b_scratch/lpc1/3DayLifetime/aman30/RunD.root");
    if (!f || !f->IsOpen()) {
      f = new TFile("file:/uscmst1b_scratch/lpc1/3DayLifetime/aman30/RunD.root");
    }
    TDirectory * dir = (TDirectory*)f->Get("/uscmst1b_scratch/lpc1/3DayLifetime/aman30/RunD.root:/HcalHBHEMuonAnalyzer");
    dir->GetObject("TREE",tree);
  }
  Init(tree);
  
  //Now book histograms
  BookHistograms(outFileName);
}

HBHEMuonOfflineAnalyzer::~HBHEMuonOfflineAnalyzer() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t HBHEMuonOfflineAnalyzer::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Int_t HBHEMuonOfflineAnalyzer::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t HBHEMuonOfflineAnalyzer::LoadTree(Long64_t entry) {
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

void HBHEMuonOfflineAnalyzer::Init(TTree *tree) {
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
  PF_Muon = 0;
  Global_Muon = 0;
  Tracker_muon = 0;
  hcal_3into3 = 0;
  hcal_1x1 = 0;
  hcal_detID = 0;
  hcal_edepth1 = 0;
  hcal_edepth2 = 0;
  hcal_edepth3 = 0;
  hcal_edepth4 = 0;
  hcal_edepthHot1 = 0;
  hcal_edepthHot2 = 0;
  hcal_edepthHot3 = 0;
  hcal_edepthHot4 = 0;
  TrackerLayer = 0;
  matchedId = 0;
  innerTrack = 0;
  innerTrackpt = 0;
  innerTracketa = 0;
  innerTrackphi = 0;
  MatchedStat = 0;
  GlobalTrckPt = 0;
  GlobalTrckEta = 0;
  GlobalTrckPhi = 0;
  NumPixelLayers = 0;
  chiTracker = 0;
  DxyTracker = 0;
  DzTracker = 0;
  OuterTrack = 0;
  OuterTrackPt = 0;
  OuterTrackEta = 0;
  OuterTrackPhi = 0;
  OuterTrackHits = 0;
  OuterTrackRHits = 0;
  OuterTrackChi = 0;
  GlobalTrack = 0;
  GlobTrack_Chi = 0;
  Global_Muon_Hits = 0;
  MatchedStations = 0;
  Global_Track_Pt = 0;
  Global_Track_Eta = 0;
  Global_Track_Phi = 0;
  Tight_LongitudinalImpactparameter = 0;
  Tight_TransImpactparameter = 0;
  InnerTrackPixelHits = 0;
  IsolationR04 = 0;
  IsolationR03 = 0;
  hcal_cellHot = 0;
  ecal_3into3 = 0;
  ecal_3x3 = 0;
  ecal_detID = 0;
  ehcal_detID = 0;
  tracker_3into3 = 0;
  activeLength = 0;
  hltresults = 0;
  all_triggers = 0;

  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("Event_No", &Event_No, &b_Event_No);
  fChain->SetBranchAddress("Run_No", &Run_No, &b_Run_No);
  fChain->SetBranchAddress("LumiNumber", &LumiNumber, &b_LumiNumber);
  fChain->SetBranchAddress("BXNumber", &BXNumber, &b_BXNumber);
  fChain->SetBranchAddress("pt_of_muon", &pt_of_muon, &b_pt_of_muon);
  fChain->SetBranchAddress("eta_of_muon", &eta_of_muon, &b_eta_of_muon);
  fChain->SetBranchAddress("phi_of_muon", &phi_of_muon, &b_phi_of_muon);
  fChain->SetBranchAddress("energy_of_muon", &energy_of_muon, &b_energy_of_muon);
  fChain->SetBranchAddress("p_of_muon", &p_of_muon, &b_p_of_muon);
  fChain->SetBranchAddress("PF_Muon", &PF_Muon, &b_PF_Muon);
  fChain->SetBranchAddress("Global_Muon", &Global_Muon, &b_Global_Muon);
  fChain->SetBranchAddress("Tracker_muon", &Tracker_muon, &b_Tracker_muon);
  fChain->SetBranchAddress("hcal_3into3", &hcal_3into3, &b_hcal_3into3);
  fChain->SetBranchAddress("hcal_1x1", &hcal_1x1, &b_hcal_1x1);
  fChain->SetBranchAddress("hcal_detID", &hcal_detID, &b_hcal_detID);
  fChain->SetBranchAddress("hcal_edepth1", &hcal_edepth1, &b_hcal_edepth1);
  fChain->SetBranchAddress("hcal_edepth2", &hcal_edepth2, &b_hcal_edepth2);
  fChain->SetBranchAddress("hcal_edepth3", &hcal_edepth3, &b_hcal_edepth3);
  fChain->SetBranchAddress("hcal_edepth4", &hcal_edepth4, &b_hcal_edepth4);
  fChain->SetBranchAddress("hcal_edepthHot1", &hcal_edepthHot1, &b_hcal_edepthHot1);
  fChain->SetBranchAddress("hcal_edepthHot2", &hcal_edepthHot2, &b_hcal_edepthHot2);
  fChain->SetBranchAddress("hcal_edepthHot3", &hcal_edepthHot3, &b_hcal_edepthHot3);
  fChain->SetBranchAddress("hcal_edepthHot4", &hcal_edepthHot4, &b_hcal_edepthHot4);
  fChain->SetBranchAddress("TrackerLayer", &TrackerLayer, &b_TrackerLayer);
  fChain->SetBranchAddress("matchedId", &matchedId, &b_matchedId);
  fChain->SetBranchAddress("innerTrack", &innerTrack, &b_innerTrack);
  fChain->SetBranchAddress("innerTrackpt", &innerTrackpt, &b_innerTrackpt);
  fChain->SetBranchAddress("innerTracketa", &innerTracketa, &b_innerTracketa);
  fChain->SetBranchAddress("innerTrackphi", &innerTrackphi, &b_innerTrackphi);
  fChain->SetBranchAddress("MatchedStat", &MatchedStat, &b_MatchedStat);
  fChain->SetBranchAddress("GlobalTrckPt", &GlobalTrckPt, &b_GlobalTrckPt);
  fChain->SetBranchAddress("GlobalTrckEta", &GlobalTrckEta, &b_GlobalTrckEta);
  fChain->SetBranchAddress("GlobalTrckPhi", &GlobalTrckPhi, &b_GlobalTrckPhi);
  fChain->SetBranchAddress("NumPixelLayers", &NumPixelLayers, &b_NumPixelLayers);
  fChain->SetBranchAddress("chiTracker", &chiTracker, &b_chiTracker);
  fChain->SetBranchAddress("DxyTracker", &DxyTracker, &b_DxyTracker);
  fChain->SetBranchAddress("DzTracker", &DzTracker, &b_DzTracker);
  fChain->SetBranchAddress("OuterTrack", &OuterTrack, &b_OuterTrack);
  fChain->SetBranchAddress("OuterTrackPt", &OuterTrackPt, &b_OuterTrackPt);
  fChain->SetBranchAddress("OuterTrackEta", &OuterTrackEta, &b_OuterTrackEta);
  fChain->SetBranchAddress("OuterTrackPhi", &OuterTrackPhi, &b_OuterTrackPhi);
  fChain->SetBranchAddress("OuterTrackHits", &OuterTrackHits, &b_OuterTrackHits);
  fChain->SetBranchAddress("OuterTrackRHits", &OuterTrackRHits, &b_OuterTrackRHits);
  fChain->SetBranchAddress("OuterTrackChi", &OuterTrackChi, &b_OuterTrackChi);
  fChain->SetBranchAddress("GlobalTrack", &GlobalTrack, &b_GlobalTrack);
  fChain->SetBranchAddress("GlobTrack_Chi", &GlobTrack_Chi, &b_GlobTrack_Chi);
  fChain->SetBranchAddress("Global_Muon_Hits", &Global_Muon_Hits, &b_Global_Muon_Hits);
  fChain->SetBranchAddress("MatchedStations", &MatchedStations, &b_MatchedStations);
  fChain->SetBranchAddress("Global_Track_Pt", &Global_Track_Pt, &b_Global_Track_Pt);
  fChain->SetBranchAddress("Global_Track_Eta", &Global_Track_Eta, &b_Global_Track_Eta);
  fChain->SetBranchAddress("Global_Track_Phi", &Global_Track_Phi, &b_Global_Track_Phi);
  fChain->SetBranchAddress("Tight_LongitudinalImpactparameter", &Tight_LongitudinalImpactparameter, &b_Tight_LongitudinalImpactparameter);
  fChain->SetBranchAddress("Tight_TransImpactparameter", &Tight_TransImpactparameter, &b_Tight_TransImpactparameter);
  fChain->SetBranchAddress("InnerTrackPixelHits", &InnerTrackPixelHits, &b_InnerTrackPixelHits);
  fChain->SetBranchAddress("IsolationR04", &IsolationR04, &b_IsolationR04);
  fChain->SetBranchAddress("IsolationR03", &IsolationR03, &b_IsolationR03);
  fChain->SetBranchAddress("hcal_cellHot", &hcal_cellHot, &b_hcal_cellHot);
  fChain->SetBranchAddress("ecal_3into3", &ecal_3into3, &b_ecal_3into3);
  fChain->SetBranchAddress("ecal_3x3", &ecal_3x3, &b_ecal_3x3);
  fChain->SetBranchAddress("ecal_detID", &ecal_detID, &b_ecal_detID);
  fChain->SetBranchAddress("ehcal_detID", &ehcal_detID, &b_ehcal_detID);
  fChain->SetBranchAddress("tracker_3into3", &tracker_3into3, &b_tracker_3into3);
  fChain->SetBranchAddress("activeLength", &activeLength, &b_activeLength);
  fChain->SetBranchAddress("hltresults", &hltresults, &b_hltresults);
  fChain->SetBranchAddress("all_triggers", &all_triggers, &b_all_triggers);
  Notify();
}

void HBHEMuonOfflineAnalyzer::Loop() {

  //declarations
  if (fChain == 0) return;
  
  Long64_t nentries = fChain->GetEntriesFast();
  
  if(debug_) std::cout << "nevent = " << nentries << std::endl;

  Long64_t nbytes = 0, nb = 0;

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;

    for (unsigned int ml = 0;ml< pt_of_muon->size();ml++) {

      if(debug_) std::cout << "ecal_det_id " << ecal_detID->at(ml) << std::endl;
      
      int typeEcal, etaXEcal, phiYEcal, zsideEcal, planeEcal, stripEcal;
      etaPhiEcal(ecal_detID->at(ml),typeEcal,zsideEcal,etaXEcal,phiYEcal,planeEcal,stripEcal);
      double etaEcal = (etaXEcal-0.5)*zsideEcal;
      double phiEcal = phiYEcal-0.5;

      if(debug_) std::cout << "hcal_det_id " << std::hex << hcal_detID->at(ml) << std::dec;

      int etaHcal, phiHcal, depthHcal;
      etaPhiHcal(hcal_detID->at(ml),etaHcal,phiHcal,depthHcal);

      int eta = (etaHcal > 0) ? etaHcal-1 : -(1+etaHcal);
      
      double etaXHcal = (etaHcal > 0) ? etaHcal-0.5 : etaHcal+0.5;
      
      if(debug_)  std::cout<<"phiHcal"<<phiHcal;

      double phiYHcal = (phiHcal-0.5);

      if(debug_) std::cout<<"phiYHcal"<<phiYHcal<<std::endl; 

      for (int cut=0; cut<3; ++cut) {
	bool select(false);
	if      (cut == 0) select = tightMuon(ml);
	else if (cut == 1) select = SoftMuon(ml);
	else               select = LooseMuon(ml);
	

	if (select) {
	  //	  h_P_Muon[cut]->Fill(p_of_muon->at(ml));
	  h_Pt_Muon[cut]->Fill(pt_of_muon->at(ml));
	  h_Eta_Muon[cut]->Fill(eta_of_muon->at(ml));
	  h_Phi_Muon[cut]->Fill(phi_of_muon->at(ml));
	  h_PF_Muon[cut]->Fill(PF_Muon->at(ml));
	  h_GlobTrack_Chi[cut]->Fill(GlobTrack_Chi->at(ml));
	  h_Global_Muon_Hits[cut]->Fill(Global_Muon_Hits->at(ml));
	  h_MatchedStations[cut]->Fill(MatchedStations->at(ml));
	  h_Tight_TransImpactparameter[cut]->Fill(Tight_TransImpactparameter->at(ml));
	  h_Tight_LongitudinalImpactparameter[cut]->Fill(Tight_LongitudinalImpactparameter->at(ml));
	  h_InnerTrackPixelHits[cut]->Fill(InnerTrackPixelHits->at(ml));
	  h_TrackerLayer[cut]->Fill(TrackerLayer->at(ml));
	  h_IsolationR04[cut]->Fill(IsolationR04->at(ml));
	  h_Global_Muon[cut]->Fill(Global_Muon->at(ml));
	  
	  h_TransImpactParameter[cut]->Fill(Tight_TransImpactparameter->at(ml));
	  h_LongImpactParameter[cut]->Fill(Tight_LongitudinalImpactparameter->at(ml));
	  
	  //in Phi Bins
	  if(((phi_of_muon->at(ml)) >= -1.5) || ((phi_of_muon->at(ml)) <= 0.5)) {
	    h_TransImpactParameterBin1[cut]->Fill(Tight_TransImpactparameter->at(ml));
	    h_LongImpactParameterBin1[cut]->Fill(Tight_LongitudinalImpactparameter->at(ml));
	    h_2D_Bin1[cut]->Fill(Tight_TransImpactparameter->at(ml),Tight_LongitudinalImpactparameter->at(ml));
	  }
	  
	  if((phi_of_muon->at(ml) > 0.5) || (phi_of_muon->at(ml) < -1.5)) {
	    h_TransImpactParameterBin2[cut]->Fill(Tight_TransImpactparameter->at(ml));
	    h_LongImpactParameterBin2[cut]->Fill(Tight_LongitudinalImpactparameter->at(ml));
	    h_2D_Bin2[cut]->Fill(Tight_TransImpactparameter->at(ml),Tight_LongitudinalImpactparameter->at(ml));
	  }


	  h_ecal_energy[cut]->Fill(ecal_3into3->at(ml));
	  h_3x3_ecal[cut]->Fill(ecal_3x3->at(ml));
	  h_Eta_ecal[cut]->Fill(eta_of_muon->at(ml),ecal_3x3->at(ml));
	  h_Phi_ecal[cut]->Fill(phi_of_muon->at(ml),ecal_3x3->at(ml));
	  h_MuonHittingEcal[cut]->Fill(typeEcal);
	  if (typeEcal == 1) {
	    h_EtaX_ecal[cut]->Fill(etaEcal,ecal_3x3->at(ml));
	    h_PhiY_ecal[cut]->Fill(phiEcal,ecal_3x3->at(ml));
	  }
	  
	  h_hcal_energy[cut]->Fill(hcal_3into3->at(ml));
	  h_1x1_hcal[cut]->Fill(hcal_1x1->at(ml));
	  h_EtaX_hcal[cut]->Fill(etaXHcal,hcal_1x1->at(ml));
	  h_PhiY_hcal[cut]->Fill(phiYHcal,hcal_1x1->at(ml));
	  h_HotCell[cut]->Fill(hcal_cellHot->at(ml));
	  
	  for (int dep=0; dep<nDepths[eta]; ++dep) {
	    
	    if(debug_) std::cout<<"why on 15/2 only"<<std::endl; 
	    if(debug_) std::cout<<"dep:"<<dep<<std::endl;
	    
	    int PHI = phiHcal;
	    
	    double en1(-9999), en2(-9999);
	    if (dep == 0) {
	      en1 = hcal_edepth1->at(ml);
	      en2 = hcal_edepthHot1->at(ml);
	    } else if (dep == 1) {
	      en1 = hcal_edepth2->at(ml);
	      en2 = hcal_edepthHot2->at(ml);
	      if(debug_) std::cout<<"problem here.. lets see if it got printed"<<std::endl;
	    } else if (dep == 2) {
	      en1 = hcal_edepth3->at(ml);
	      en2 = hcal_edepthHot3->at(ml);
	    } else if (dep == 3) {
	      en1 = hcal_edepth4->at(ml);
	      en2 = hcal_edepthHot4->at(ml);
	      if(debug_) std::cout<<"Hello in 4"<<std::endl;
	    }
	    
	    if(debug_) std::cout<<" Debug2"<<std::endl;
	    if(debug_) std::cout<<"ok1"<<en1<<std::endl;
	    if(debug_) std::cout<<"ok2"<<en2<<std::endl;
	    
	    bool ok1 = (en1 > -9999);
	    bool ok2 = (en2 > -9999);
	    
	    if(debug_) std::cout<<"Before Index"<<std::endl; 
	    
	    int ind = (etaHcal > 0) ? indxEta[eta][dep][PHI] : 1+indxEta[eta][dep][PHI];
	    
	    if(debug_) std::cout<<"ieta"<<eta<<"depth"<<dep<<"indxEta[eta][dep]:"<<indxEta[eta][dep]<<std::endl;
	    
	    if(debug_) std::cout<<"index showing eta,depth:"<<ind<<std::endl;
	    
	    if(debug_) std::cout << "etaHcal: " << etaHcal << " eta " << eta << " dep " << dep << " indx " << ind << std::endl;
	    if(!(matchedId->at(ml))) continue;
	    if (ok1) {
	      if(debug_) std::cout<<"enter ok1"<<std::endl;
	      
	      if (hcal_cellHot->at(ml)==1) {
		if(en2 > 0) {
		  h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[cut][ind]->Fill(en2/activeLength->at(ml));
		}
		if(debug_) std::cout<<"enter hot cell"<<std::endl;
	      }
	    }
	    
	    if (ok2) {
	      if(debug_) std::cout<<"enter ok2"<<std::endl;
	      if (hcal_cellHot->at(ml)!=1) {
	      }
	    }
	    
	    if(debug_) std::cout<<"ETA \t"<<eta<<"DEPTH \t"<<dep<<std::endl;
	    
	  }
	}
      }
    }
  }
  close();
}

Bool_t HBHEMuonOfflineAnalyzer::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void HBHEMuonOfflineAnalyzer::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

void HBHEMuonOfflineAnalyzer::BookHistograms(const char* fname) {
  output_file = TFile::Open(fname,"RECREATE");
  std::string type[]={"tight","soft","loose"};
  char name[128], title[500];
  
  std::cout<<"BookHistograms"<<std::endl;
  int  nDepth[29]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,1,2,2,2,2,2,2,2,2,2,3,3,2};
  int  nDepthPhi[29]={72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,36,36,36,36,36,36,36,36,36};

  // for 2019 scenario multi depth segmentation
  //    int  nDepth[29]={3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5,5,5};
  nHist = 0;
  for (int eta=0; eta<29; ++eta) {

    nDepthsPhi[eta] = nDepthPhi[eta];
    if (modeLHC != 0) {
      nDepths[eta] = nDepth[eta];
    } else if (eta<15) {
      nDepths[eta] = maxDepthHB_;
    } else if (eta == 15) {
      nDepths[eta] = 3;   
    } else if (eta > 15 && eta < 29) {
      nDepths[eta] = maxDepthHE_;
    }

    if(debug_) std::cout<<"Eta:"<<eta<<" nDepths[eta]"<<nDepths[eta]<<std::endl;
    for (int depth=0; depth<nDepths[eta]; ++depth) {
      if(debug_) std::cout<<"Eta:"<<eta<<"Depth:"<<depth<<std::endl;
      for (int PHI=0;  PHI<nDepthsPhi[eta]; ++PHI) {
	indxEta[eta][depth][PHI] = nHist;
	nHist += 2;
      }
    }
  }
  for (int i=0; i<1; ++i) {
    sprintf (name,  "h_Pt_Muon_%s", type[i].c_str());
    sprintf (title, "p_{T} of %s muons (GeV)", type[i].c_str());
    h_Pt_Muon[i]  = new TH1D(name, title,100,0,200);
    
    sprintf (name,  "h_Eta_Muon_%s", type[i].c_str());
    sprintf (title, "#eta of %s muons", type[i].c_str());
    h_Eta_Muon[i] = new TH1D(name, title,50,-2.5,2.5);
    
    sprintf (name,  "h_Phi_Muon_%s", type[i].c_str());
    sprintf (title, "#phi of %s muons", type[i].c_str());
    h_Phi_Muon[i] = new TH1D(name, title,100,-3.1415926,3.1415926);
    
    sprintf (name,  "h_P_Muon_%s", type[i].c_str());
    sprintf (title, "p of %s muons (GeV)", type[i].c_str());
    h_P_Muon[i]   = new TH1D(name, title,100,0,200);
    
    sprintf (name,  "h_PF_Muon_%s", type[i].c_str());
    sprintf (title, "PF %s muons (GeV)", type[i].c_str());
    h_PF_Muon[i]   = new TH1D(name, title,2,0,2);
    
    sprintf (name,  "h_Global_Muon_Chi2_%s", type[i].c_str());
    sprintf (title, "Chi2 Global %s muons (GeV)", type[i].c_str());
    h_GlobTrack_Chi[i]  = new TH1D(name, title,15,0,15);
    
    sprintf (name,  "h_Global_Muon_Hits_%s", type[i].c_str());
    sprintf (title, "Global Hits %s muons (GeV)", type[i].c_str());
    h_Global_Muon_Hits[i] = new TH1D(name, title,10,0,10) ;
    
    sprintf (name,  "h_Matched_Stations_%s", type[i].c_str());
    sprintf (title, "Matched Stations %s muons (GeV)", type[i].c_str());
    h_MatchedStations[i] = new TH1D(name, title,10,0,10);

    sprintf (name,  "h_Transverse_ImpactParameter_%s", type[i].c_str());
    sprintf (title,  "Transverse_ImpactParameter of %s muons (GeV)", type[i].c_str());
    h_Tight_TransImpactparameter[i] = new TH1D(name, title,50,0,10);

    sprintf (name,  "h_Longitudinal_ImpactParameter_%s", type[i].c_str());
    sprintf (title,  "Longitudinal_ImpactParameter of %s muons (GeV)", type[i].c_str());
    h_Tight_LongitudinalImpactparameter[i] = new TH1D(name, title,20,0,10); 
    
    sprintf (name,  "h_InnerTrack_PixelHits_%s", type[i].c_str());
    sprintf (title,  "InnerTrack_PixelHits of %s muons (GeV)", type[i].c_str());
    h_InnerTrackPixelHits[i]= new TH1D(name, title,20,0,20);

    sprintf (name,  "h_TrackLayers_%s", type[i].c_str());
    sprintf (title,  "No. of Tracker Layers of %s muons (GeV)", type[i].c_str());
    h_TrackerLayer[i]= new TH1D(name, title,20,0,20);;

    sprintf (name,  "h_IsolationR04_%s", type[i].c_str());
    sprintf (title,  "IsolationR04 %s muons (GeV)", type[i].c_str());
    h_IsolationR04[i] = new TH1D(name, title,45,0,5);;
    
    sprintf (name,  "h_Global_Muon_%s", type[i].c_str());
    sprintf (title, "Global %s muons (GeV)", type[i].c_str());
    h_Global_Muon[i]= new TH1D(name, title,2,0,2);

    sprintf (name,  "h_TransImpactParameter_%s", type[i].c_str());
    sprintf (title, "TransImpactParameter of %s muons (GeV)", type[i].c_str());
    h_TransImpactParameter[i]   = new TH1D(name, title,100,0,0.5);

    sprintf (name,  "h_TransImpactParameterBin1_%s", type[i].c_str());
    sprintf (title, "TransImpactParameter of %s muons (GeV) in -1.5 <= #phi <= 0.5", type[i].c_str());
    h_TransImpactParameterBin1[i]   = new TH1D(name, title,100,0,0.5);

    sprintf (name,  "h_TransImpactParameterBin2_%s", type[i].c_str());
    sprintf (title, "TransImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
    h_TransImpactParameterBin2[i]   = new TH1D(name, title,100,0,0.5);
    //
    sprintf (name,  "h_LongImpactParameter_%s", type[i].c_str());
    sprintf (title, "LongImpactParameter of %s muons (GeV)", type[i].c_str());
    h_LongImpactParameter[i]   = new TH1D(name, title,100,0,30);

    sprintf (name,  "h_LongImpactParameterBin1_%s", type[i].c_str());
    sprintf (title, "LongImpactParameter of %s muons (GeV) in -1.5 <= #phi <= 0.5", type[i].c_str());
    h_LongImpactParameterBin1[i]   = new TH1D(name, title,100,0,30);

    sprintf (name,  "h_LongImpactParameterBin2_%s", type[i].c_str());
    sprintf (title, "LongImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
    h_LongImpactParameterBin2[i]   = new TH1D(name, title,100,0,30);

    sprintf (name,  "h_2D_Bin1_%s", type[i].c_str());
    sprintf (title, "Trans/Long ImpactParameter of %s muons (GeV) in -1.5 <= #phi< 0.5 ", type[i].c_str());
    h_2D_Bin1[i] = new TH2D(name, title, 100,0,0.5,100,0,30);

    sprintf (name,  "h_2D_Bin2_%s", type[i].c_str());
    sprintf (title, "Trans/Long ImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
    h_2D_Bin2[i] = new TH2D(name, title, 100,0,0.5,100,0,30);

    sprintf (name,  "h_ecal_energy_%s", type[i].c_str());
    sprintf (title, "ECAL energy for %s muons", type[i].c_str());
    h_ecal_energy[i] = new TH1D(name, title,1000,-10.0,90.0);

    sprintf (name,  "h_hcal_energy_%s", type[i].c_str());
    sprintf (title, "HCAL energy for %s muons", type[i].c_str());
    h_hcal_energy[i] = new TH1D(name, title,500,-10.0,90.0);

    sprintf (name,  "h_3x3_ecal_%s", type[i].c_str());
    sprintf (title, "ECAL energy in 3x3 for %s muons", type[i].c_str());
    h_3x3_ecal[i] = new TH1D(name, title,1000,-10.0,90.0);

    sprintf (name,  "h_1x1_hcal_%s", type[i].c_str());
    sprintf (title, "HCAL energy in 1x1 for %s muons", type[i].c_str());
    h_1x1_hcal[i] = new TH1D(name, title,500,-10.0,90.0);

    sprintf (name,  "h_EtaX_hcal_%s", type[i].c_str());
    sprintf (title, "HCAL energy as a function of i#eta for %s muons", type[i].c_str());
    h_EtaX_hcal[i] = new TProfile(name, title,60,-30.0,30.0);

    sprintf (name,  "h_PhiY_hcal_%s", type[i].c_str());
    sprintf (title, "HCAL energy as a function of i#phi for %s muons", type[i].c_str());
    h_PhiY_hcal[i] = new TProfile(name, title,72,0,72);

    sprintf (name,  "h_EtaX_ecal_%s", type[i].c_str());
    sprintf (title, "EB energy as a function of i#eta for %s muons", type[i].c_str());
    h_EtaX_ecal[i] = new TProfile(name, title,170,-85.0,85.0);

    sprintf (name,  "h_PhiY_ecal_%s", type[i].c_str());
    sprintf (title, "EB energy as a function of i#phi for %s muons", type[i].c_str());
    h_PhiY_ecal[i] = new TProfile(name, title,360,0,360);

    sprintf (name,  "h_Eta_ecal_%s", type[i].c_str());
    sprintf (title, "ECAL energy as a function of #eta for %s muons", type[i].c_str());
    h_Eta_ecal[i] = new TProfile(name, title,100,-2.5,2.5);
    
    sprintf (name,  "h_Phi_ecal_%s", type[i].c_str());
    sprintf (title, "ECAL energy as a function of #phi for %s muons", type[i].c_str());
    h_Phi_ecal[i] = new TProfile(name, title,100,-3.1415926,3.1415926);
    
    sprintf (name,  "h_MuonHittingEcal_%s", type[i].c_str());
    sprintf (title, "%s muons hitting ECAL", type[i].c_str());
    h_MuonHittingEcal[i] = new TH1D(name, title,100,0,5.0);

    sprintf (name,  "h_HotCell_%s", type[i].c_str());
    sprintf (title, "Hot cell for %s muons", type[i].c_str());
    h_HotCell[i] = new TH1D(name, title,100,0,2);

    for (int eta=0; eta<29; ++eta) {
      for (int depth=0; depth<nDepths[eta]; ++depth) {
	for (int PHI=0; PHI<nDepthsPhi[eta]; ++PHI) {
	  
	  std::cout<<"eta:"<<eta<<"depth:"<<depth<<"PHI:"<<PHI<<std::endl;
	  int ih = indxEta[eta][depth][PHI];
	  std::cout<<"ih:"<<ih<<std::endl; 
	  
	  sprintf (name,  "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell_ByActiveLength", (eta+1), (depth+1), PHI, type[i].c_str());
	  sprintf (title, "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi = %d) for extrapolated %s muons (Hot Cell) divided by Active Length", (eta+1), (depth+1), PHI, type[i].c_str());
	  h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih] = new TH1D(name, title,4000,0.0,10.0); 

	  ih++;
	  sprintf (name,  "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell_ByActiveLength", -(eta+1), (depth+1),PHI, type[i].c_str());
	  sprintf (title, "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi=%d) for extrapolated %s muons (Hot Cell) divided by Active Length", -(eta+1), (depth+1), PHI, type[i].c_str());
	  h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[i][ih] = new TH1D(name, title,4000,0.0,10.0);
	}
      }
    } 
  }
}

bool HBHEMuonOfflineAnalyzer::LooseMuon(unsigned int ml){
  if (PF_Muon->at(ml)) {
    if (Global_Muon->at(ml) && Tracker_muon->at(ml)) {
      if (pt_of_muon->at(ml)>20.) {
	if (IsolationR04->at(ml) < 0.12) { 
	  return true;
	}
      }
    }
  }
  return false;
} 

bool HBHEMuonOfflineAnalyzer::SoftMuon(unsigned int ml){
  if (innerTrack->at(ml)){
    if (pt_of_muon->at(ml) > 20.){
      if (TrackerLayer->at(ml)>5) {
	if (innerTrack->at(ml) && NumPixelLayers->at(ml)> 1) {
	  if (chiTracker->at(ml) < 1.8 && DxyTracker->at(ml) <3 && DzTracker->at(ml) <30) {
	    if (IsolationR04->at(ml) < 0.12) {
	      return true;
	    }
	  }
	}
      }
    }
  } 
  return false;
}

bool HBHEMuonOfflineAnalyzer::tightMuon(unsigned int ml) {
  if (PF_Muon->at(ml) && (Global_Muon->at(ml))) {
    //    if(debug_) std::cout<<"global and PF Muon"<<std::endl;
    if (GlobalTrack->at(ml)) {
      if (pt_of_muon->at(ml) > 20.) {
	if (GlobTrack_Chi->at(ml) < 10 && Global_Muon_Hits->at(ml) >0 && MatchedStations->at(ml)>1){
	  if (Tight_TransImpactparameter->at(ml) < 0.2 && Tight_LongitudinalImpactparameter->at(ml) < 0.5 ){
	    if (InnerTrackPixelHits->at(ml) > 0 && TrackerLayer->at(ml) > 5){
	      if (IsolationR04->at(ml) < 0.12) { 
		return true;   
	      }
	    }
	  }
	}
      }
    }
  }
  return false;
}

void HBHEMuonOfflineAnalyzer::etaPhiHcal(unsigned int detId, int &eta, int &phi, int &depth) {
  int zside, etaAbs;
  if ((detId&0x1000000)==0) {
    zside  = (detId&0x2000)?(1):(-1);
    etaAbs = (detId>>7)&0x3F;
    phi    = detId&0x7F;
    depth  = (detId>>14)&0x1F;
  } else {
    zside  = (detId&0x80000)?(1):(-1);
    etaAbs = (detId>>10)&0x1FF;
    phi    = detId&0x3FF;
    depth  = (detId>>20)&0xF;
  }
  eta    = etaAbs*zside;
}

void HBHEMuonOfflineAnalyzer::etaPhiEcal(unsigned int detId, int& type, int& zside,
				 int& etaX, int& phiY, int& plane, int& strip) {

  type = ((detId>>25)&0x7);
  // std::cout<<"type"<<type<<std::endl; 
  plane = strip = 0;
  if (type==1) {
    //Ecal Barrel
    zside = (detId&0x10000)?(1):(-1);
    etaX  = (detId>>9)&0x7F;
    phiY  =  detId&0x1FF;
  } else if (type==2) {
    zside = (detId&0x4000)?(1):(-1);
    etaX  = (detId>>7)&0x7F;
    phiY  = (detId&0x7F);
  } else if (type==3) {
    zside = (detId&0x80000)?(1):(-1);
    etaX  = (detId>>6)&0x3F;
    /** get the sensor iy */
    phiY  = (detId>>12)&0x3F;
    /** get the strip */
    plane = ((detId>>18)&0x1)+1;
    strip = detId&0x3F;
  } else {
    zside = etaX = phiY = 0;
  }
}


void HBHEMuonOfflineAnalyzer::calculateP(double pt, double eta, double& pM) {
  pM = (pt*cos(2*(1/atan(exp(eta)))));
}

void HBHEMuonOfflineAnalyzer::close() {
  output_file->cd();
  std::cout << "file yet to be Written" << std::endl;
  output_file->Write();
  std::cout << "file Written" << std::endl;
  output_file->Close();
  std::cout << "now doing return" << std::endl;
}
