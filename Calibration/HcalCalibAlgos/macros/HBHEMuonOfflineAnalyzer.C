///////////////////////////////////////////////////////////////////////////////
//
//  HBHEMuonOfflineAnalyzer h1(tree, outfile, rcorFile, flag, mode, maxDHB,
//                             maxDHE, cutMu, cutP, nevMax, over, runLo,
//                             runHi, etaMin, etaMax, debug);
//  HBHEMuonOfflineAnalyzer h1(infile, outfile, rcorFile, flag, mode, maxDHB,
//                             maxDHE, cutMu, cutP, nevMax, over, runLo,
//                             runHi, etaMin, etaMax, debug);
//   h1.Loop()
//
//   tree       TTree*       Pointer to the tree chain
//   infile     const char*  Name of the input file
//   outfile    const char*  Name of the output file
//                           (dyll_PU20_25_output_10.root)
//   rcorFile   consr char*  name of the text file having the correction factors
//                           as a function of run numbers to be used for raddam
//                           correction (default="", no corr.)
//   flag       int          Flag of 2 digits ("to"): to where "o" decides if
//                           corrected (1) or default (0) energy to be used;
//                           "t" decides if all depths to be merged (1) or not
//                           (0) (default is 0)
//   mode       int          Geometry file used 0:(defined by maxDHB/HE);
//                           1 (Run 1; valid till 2016); 2 (Run 2; 2018);
//                           3 (Run 3; post LS2); 4 (2017 Plan 1);
//                           5 (Run 4; post LS3); default (3)
//   maxDHB     int          Maximum number of depths for HB (4)
//   maxDHE     int          Maximum number of depths for HE (7)
//   cutMu      int          Selection of muon type:
//                           (tight:0; medium:1; loose:2) default (0)
//   cutP       float        Minimum muon momentum; default (10)
//   nevMax     int          Maximum # oe entries to be processed; -1 means
//                           all entries (-1)
//   over       int          Override some of the selection
//                           (0: not to override; 1: override) default (0)
//   runLO      int          Minimum run number (1)
//   runHI      int          Maximum run number (99999999)
//   etaMin     int          Minimum (absolute) eta value (1)
//   etaMax     int          Maximum (absolute) eta value (29)
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
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
#include <TString.h>
#include <TTree.h>

class HBHEMuonOfflineAnalyzer {
public:
  TChain *fChain;  //!pointer to the analyzed TTree/TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  // Fixed size dimensions of array or collections stored in the TTree if any.
  // Declaration of leaf types

  UInt_t Event_No;
  UInt_t Run_No;
  UInt_t LumiNumber;
  UInt_t BXNumber;
  UInt_t GoodVertex;
  bool PF_Muon;
  bool Global_Muon;
  bool Tracker_muon;
  bool MuonIsTight;
  bool MuonIsMedium;
  double pt_of_muon;
  double eta_of_muon;
  double phi_of_muon;
  double energy_of_muon;
  double p_of_muon;
  float muon_trkKink;
  float muon_chi2LocalPosition;
  float muon_segComp;
  int TrackerLayer;
  int NumPixelLayers;
  int InnerTrackPixelHits;
  bool innerTrack;
  double chiTracker;
  double DxyTracker;
  double DzTracker;
  double innerTrackpt;
  double innerTracketa;
  double innerTrackphi;
  double tight_validFraction;
  bool OuterTrack;
  double OuterTrackPt;
  double OuterTrackEta;
  double OuterTrackPhi;
  double OuterTrackChi;
  int OuterTrackHits;
  int OuterTrackRHits;
  bool GlobalTrack;
  double GlobalTrckPt;
  double GlobalTrckEta;
  double GlobalTrckPhi;
  int Global_Muon_Hits;
  int MatchedStations;
  double GlobTrack_Chi;
  double Tight_LongitudinalImpactparameter;
  double Tight_TransImpactparameter;
  double IsolationR04;
  double IsolationR03;
  double ecal_3into3;
  double hcal_3into3;
  double tracker_3into3;
  bool matchedId;
  bool hcal_cellHot;
  double ecal_3x3;
  double hcal_1x1;
  unsigned int ecal_detID;
  unsigned int hcal_detID;
  unsigned int ehcal_detID;
  int hcal_ieta;
  int hcal_iphi;
  double hcal_edepth1;
  double hcal_activeL1;
  double hcal_edepthHot1;
  double hcal_activeHotL1;
  double hcal_edepthCorrect1;
  double hcal_edepthHotCorrect1;
  double hcal_cdepthHot1;
  double hcal_cdepthHotBG1;
  bool hcal_depthMatch1;
  bool hcal_depthMatchHot1;
  double hcal_edepth2;
  double hcal_activeL2;
  double hcal_edepthHot2;
  double hcal_activeHotL2;
  double hcal_edepthCorrect2;
  double hcal_edepthHotCorrect2;
  double hcal_cdepthHot2;
  double hcal_cdepthHotBG2;
  bool hcal_depthMatch2;
  bool hcal_depthMatchHot2;
  double hcal_edepth3;
  double hcal_activeL3;
  double hcal_edepthHot3;
  double hcal_activeHotL3;
  double hcal_edepthCorrect3;
  double hcal_edepthHotCorrect3;
  double hcal_cdepthHot3;
  double hcal_cdepthHotBG3;
  bool hcal_depthMatch3;
  bool hcal_depthMatchHot3;
  double hcal_edepth4;
  double hcal_activeL4;
  double hcal_edepthHot4;
  double hcal_activeHotL4;
  double hcal_edepthCorrect4;
  double hcal_edepthHotCorrect4;
  double hcal_cdepthHot4;
  double hcal_cdepthHotBG4;
  bool hcal_depthMatch4;
  bool hcal_depthMatchHot4;
  double hcal_edepth5;
  double hcal_activeL5;
  double hcal_edepthHot5;
  double hcal_activeHotL5;
  double hcal_edepthCorrect5;
  double hcal_edepthHotCorrect5;
  double hcal_cdepthHot5;
  double hcal_cdepthHotBG5;
  bool hcal_depthMatch5;
  bool hcal_depthMatchHot5;
  double hcal_edepth6;
  double hcal_activeL6;
  double hcal_edepthHot6;
  double hcal_activeHotL6;
  double hcal_edepthCorrect6;
  double hcal_edepthHotCorrect6;
  double hcal_cdepthHot6;
  double hcal_cdepthHotBG6;
  bool hcal_depthMatch6;
  bool hcal_depthMatchHot6;
  double hcal_edepth7;
  double hcal_activeL7;
  double hcal_edepthHot7;
  double hcal_activeHotL7;
  double hcal_edepthCorrect7;
  double hcal_edepthHotCorrect7;
  double hcal_cdepthHot7;
  double hcal_cdepthHotBG7;
  bool hcal_depthMatch7;
  bool hcal_depthMatchHot7;
  double activeLength;
  double activeLengthHot;
  std::vector<int> *hltresults;
  std::vector<std::string> *all_triggers;

  // List of branches
  TBranch *b_Event_No;                           //!
  TBranch *b_Run_No;                             //!
  TBranch *b_LumiNumber;                         //!
  TBranch *b_BXNumber;                           //!
  TBranch *b_GoodVertex;                         //!
  TBranch *b_PF_Muon;                            //!
  TBranch *b_Global_Muon;                        //!
  TBranch *b_Tracker_muon;                       //!
  TBranch *b_MuonIsTight;                        //!
  TBranch *b_MuonIsMedium;                       //!
  TBranch *b_pt_of_muon;                         //!
  TBranch *b_eta_of_muon;                        //!
  TBranch *b_phi_of_muon;                        //!
  TBranch *b_energy_of_muon;                     //!
  TBranch *b_p_of_muon;                          //!
  TBranch *b_muon_trkKink;                       //!
  TBranch *b_muon_chi2LocalPosition;             //!
  TBranch *b_muon_segComp;                       //!
  TBranch *b_TrackerLayer;                       //!
  TBranch *b_NumPixelLayers;                     //!
  TBranch *b_InnerTrackPixelHits;                //!
  TBranch *b_innerTrack;                         //!
  TBranch *b_chiTracker;                         //!
  TBranch *b_DxyTracker;                         //!
  TBranch *b_DzTracker;                          //!
  TBranch *b_innerTrackpt;                       //!
  TBranch *b_innerTracketa;                      //!
  TBranch *b_innerTrackphi;                      //!
  TBranch *b_tight_validFraction;                //!
  TBranch *b_OuterTrack;                         //!
  TBranch *b_OuterTrackPt;                       //!
  TBranch *b_OuterTrackEta;                      //!
  TBranch *b_OuterTrackPhi;                      //!
  TBranch *b_OuterTrackChi;                      //!
  TBranch *b_OuterTrackHits;                     //!
  TBranch *b_OuterTrackRHits;                    //!
  TBranch *b_GlobalTrack;                        //!
  TBranch *b_GlobalTrckPt;                       //!
  TBranch *b_GlobalTrckEta;                      //!
  TBranch *b_GlobalTrckPhi;                      //!
  TBranch *b_Global_Muon_Hits;                   //!
  TBranch *b_MatchedStations;                    //!
  TBranch *b_GlobTrack_Chi;                      //!
  TBranch *b_Tight_LongitudinalImpactparameter;  //!
  TBranch *b_Tight_TransImpactparameter;         //!
  TBranch *b_IsolationR04;                       //!
  TBranch *b_IsolationR03;                       //!
  TBranch *b_ecal_3into3;                        //!
  TBranch *b_hcal_3into3;                        //!
  TBranch *b_tracker_3into3;                     //!
  TBranch *b_matchedId;                          //!
  TBranch *b_hcal_cellHot;                       //!
  TBranch *b_ecal_3x3;                           //!
  TBranch *b_hcal_1x1;                           //!
  TBranch *b_hcal_detID;                         //!
  TBranch *b_ecal_detID;                         //!
  TBranch *b_ehcal_detID;                        //!
  TBranch *b_hcal_ieta;                          //!
  TBranch *b_hcal_iphi;                          //!
  TBranch *b_hcal_edepth1;                       //!
  TBranch *b_hcal_activeL1;                      //!
  TBranch *b_hcal_edepthHot1;                    //!
  TBranch *b_hcal_activeHotL1;                   //!
  TBranch *b_hcal_edepthCorrect1;                //!
  TBranch *b_hcal_edepthHotCorrect1;             //!
  TBranch *b_hcal_cdepthHot1;                    //!
  TBranch *b_hcal_cdepthHotBG1;                  //!
  TBranch *b_hcal_depthMatch1;                   //!
  TBranch *b_hcal_depthMatchHot1;                //!
  TBranch *b_hcal_edepth2;                       //!
  TBranch *b_hcal_activeL2;                      //!
  TBranch *b_hcal_edepthHot2;                    //!
  TBranch *b_hcal_activeHotL2;                   //!
  TBranch *b_hcal_edepthCorrect2;                //!
  TBranch *b_hcal_edepthHotCorrect2;             //!
  TBranch *b_hcal_cdepthHot2;                    //!
  TBranch *b_hcal_cdepthHotBG2;                  //!
  TBranch *b_hcal_depthMatch2;                   //!
  TBranch *b_hcal_depthMatchHot2;                //!
  TBranch *b_hcal_edepth3;                       //!
  TBranch *b_hcal_activeL3;                      //!
  TBranch *b_hcal_edepthHot3;                    //!
  TBranch *b_hcal_activeHotL3;                   //!
  TBranch *b_hcal_edepthCorrect3;                //!
  TBranch *b_hcal_edepthHotCorrect3;             //!
  TBranch *b_hcal_cdepthHot3;                    //!
  TBranch *b_hcal_cdepthHotBG3;                  //!
  TBranch *b_hcal_depthMatch3;                   //!
  TBranch *b_hcal_depthMatchHot3;                //!
  TBranch *b_hcal_edepth4;                       //!
  TBranch *b_hcal_activeL4;                      //!
  TBranch *b_hcal_edepthHot4;                    //!
  TBranch *b_hcal_activeHotL4;                   //!
  TBranch *b_hcal_edepthCorrect4;                //!
  TBranch *b_hcal_edepthHotCorrect4;             //!
  TBranch *b_hcal_cdepthHot4;                    //!
  TBranch *b_hcal_cdepthHotBG4;                  //!
  TBranch *b_hcal_depthMatch4;                   //!
  TBranch *b_hcal_depthMatchHot4;                //!
  TBranch *b_hcal_edepth5;                       //!
  TBranch *b_hcal_activeL5;                      //!
  TBranch *b_hcal_edepthHot5;                    //!
  TBranch *b_hcal_activeHotL5;                   //!
  TBranch *b_hcal_edepthCorrect5;                //!
  TBranch *b_hcal_edepthHotCorrect5;             //!
  TBranch *b_hcal_cdepthHot5;                    //!
  TBranch *b_hcal_cdepthHotBG5;                  //!
  TBranch *b_hcal_depthMatch5;                   //!
  TBranch *b_hcal_depthMatchHot5;                //!
  TBranch *b_hcal_edepth6;                       //!
  TBranch *b_hcal_activeL6;                      //!
  TBranch *b_hcal_edepthHot6;                    //!
  TBranch *b_hcal_activeHotL6;                   //!
  TBranch *b_hcal_edepthCorrect6;                //!
  TBranch *b_hcal_edepthHotCorrect6;             //!
  TBranch *b_hcal_cdepthHot6;                    //!
  TBranch *b_hcal_cdepthHotBG6;                  //!
  TBranch *b_hcal_depthMatch6;                   //!
  TBranch *b_hcal_depthMatchHot6;                //!
  TBranch *b_hcal_edepth7;                       //!
  TBranch *b_hcal_activeL7;                      //!
  TBranch *b_hcal_edepthHot7;                    //!
  TBranch *b_hcal_activeHotL7;                   //!
  TBranch *b_hcal_edepthCorrect7;                //!
  TBranch *b_hcal_edepthHotCorrect7;             //!
  TBranch *b_hcal_cdepthHot7;                    //!
  TBranch *b_hcal_cdepthHotBG7;                  //!
  TBranch *b_hcal_depthMatch7;                   //!
  TBranch *b_hcal_depthMatchHot7;                //!
  TBranch *b_activeLength;                       //!
  TBranch *b_activeLengthHot;                    //!
  TBranch *b_hltresults;                         //!
  TBranch *b_all_triggers;                       //!

  HBHEMuonOfflineAnalyzer(TChain *tree = 0,
                          const char *outfile = "dyll_PU20_25_output_10.root",
                          const char *rcorFileName = "",
                          int flag = 0,
                          int mode = 3,
                          int maxDHB = 4,
                          int maxDHE = 7,
                          int cutMu = 0,
                          float cutP = 5,
                          int nevMax = -1,
                          int over = 0,
                          int runLo = 1,
                          int runHi = 99999999,
                          int etaMin = 1,
                          int etaMax = 29,
                          bool debug = false);
  HBHEMuonOfflineAnalyzer(const char *infile,
                          const char *outfile = "dyll_PU20_25_output_10.root",
                          const char *rcorFileName = "",
                          int flag = 0,
                          int mode = 3,
                          int maxDHB = 4,
                          int maxDHE = 7,
                          int cutMu = 0,
                          float cutP = 5,
                          int nevMax = -1,
                          int over = 0,
                          int runLo = 1,
                          int runHi = 99999999,
                          int etaMin = 1,
                          int etaMax = 29,
                          bool debug = false);
  // mode of LHC is kept 1 for 2017 scenario as no change in depth segmentation
  // mode of LHC is 3 for 2021
  virtual ~HBHEMuonOfflineAnalyzer();

  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *tree,
                    const char *rcorFileName,
                    int flag,
                    int mode,
                    int maxDHB,
                    int maxDHE,
                    int runLo,
                    int runHi,
                    int etaMin,
                    int etaMax);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);

  bool fillChain(TChain *chain, const char *inputFileList);
  bool readCorr(const char *rcorFileName);
  void bookHistograms(const char *);
  bool getEnergy(int dep, double &enb, double &enu, double &enh, double &enc, double &chgS, double &chgB, double &actL);
  void writeHistograms();
  bool looseMuon();
  bool tightMuon();
  bool softMuon();
  bool mediumMuon2016();
  void etaPhiHcal(unsigned int detId, int &eta, int &phi, int &depth);
  void etaPhiEcal(unsigned int detId, int &type, int &zside, int &etaX, int &phiY, int &plane, int &strip);
  void calculateP(double pt, double eta, double &pM);
  void close();
  int nDepthBins(int ieta, int iphi);
  int nPhiBins(int ieta);
  float getCorr(int run, unsigned int id);
  std::vector<std::string> splitString(const std::string &);
  unsigned int getDetIdHBHE(int ieta, int iphi, int depth);
  unsigned int getDetId(int subdet, int ieta, int iphi, int depth);
  unsigned int correctDetId(const unsigned int &detId);
  void unpackDetId(unsigned int detId, int &subdet, int &zside, int &ieta, int &iphi, int &depth);

private:
  static const int maxDep_ = 7;
  static const int maxEta_ = 29;
  static const int maxPhi_ = 72;
  //3x16x72x2 + 5x4x72x2 + 5x9x36x2
  static const int maxHist_ = 20000;  //13032;
  static const unsigned int nmax_ = 10;
  int nCut_;
  const double cutP_;
  const int nevMax_;
  const bool over_, debug_;
  int modeLHC_, maxDepthHB_, maxDepthHE_, maxDepth_;
  int runLo_, runHi_, etaMin_, etaMax_;
  bool cFactor_, useCorrect_, mergeDepth_;
  int nHist, nDepths[maxEta_], nDepthsPhi[maxEta_];
  int indxEta[maxEta_][maxDep_][maxPhi_];
  TFile *output_file;
  std::map<unsigned int, float> corrFac_[nmax_];
  std::vector<int> runlow_;

  TTree *outtree_;
  int t_ieta, t_iphi, t_nvtx;
  double t_p, t_ediff;
  std::vector<double> t_ene, t_enec, t_actln, t_charge;
  std::vector<int> t_depth;

  TH1D *h_evtype, *h_Pt_Muon, *h_Eta_Muon, *h_Phi_Muon, *h_P_Muon;
  TH1D *h_PF_Muon, *h_GlobTrack_Chi, *h_Global_Muon_Hits;
  TH1D *h_MatchedStations, *h_Tight_TransImpactparameter;
  TH1D *h_Tight_LongitudinalImpactparameter, *h_InnerTrackPixelHits;
  TH1D *h_TrackerLayer, *h_IsolationR04, *h_Global_Muon;
  TH1D *h_LongImpactParameter, *h_LongImpactParameterBin1, *h_LongImpactParameterBin2;

  TH1D *h_TransImpactParameter, *h_TransImpactParameterBin1, *h_TransImpactParameterBin2;
  TH1D *h_Hot_MuonEnergy_hcal_ClosestCell[maxHist_], *h_Hot_MuonEnergy_hcal_HotCell[maxHist_],
      *h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[maxHist_], *h_HotCell_MuonEnergy_phi[maxHist_],
      *h_active_length_Fill[maxHist_], *h_p_muon_ineta[maxHist_], *h_charge_signal[maxHist_], *h_charge_bg[maxHist_];
  TH2D *h_2D_Bin1, *h_2D_Bin2;
  TH1D *h_ecal_energy, *h_hcal_energy, *h_3x3_ecal, *h_1x1_hcal;
  TH1D *h_MuonHittingEcal, *h_HotCell, *h_MuonEnergy_hcal[maxHist_];
  TH1D *h_Hot_MuonEnergy_hcal[maxHist_];
  TH2D *hcal_ietaVsEnergy;
  TProfile *h_EtaX_hcal, *h_PhiY_hcal, *h_EtaX_ecal, *h_PhiY_ecal;
  TProfile *h_Eta_ecal, *h_Phi_ecal;
  TProfile *h_MuonEnergy_eta[maxDep_], *h_MuonEnergy_phi[maxDep_], *h_MuonEnergy_muon_eta[maxDep_];
  TProfile *h_Hot_MuonEnergy_eta[maxDep_], *h_Hot_MuonEnergy_phi[maxDep_], *h_Hot_MuonEnergy_muon_eta[maxDep_];
  TProfile *h_IsoHot_MuonEnergy_eta[maxDep_], *h_IsoHot_MuonEnergy_phi[maxDep_], *h_IsoHot_MuonEnergy_muon_eta[maxDep_];
  TProfile *h_IsoWithoutHot_MuonEnergy_eta[maxDep_], *h_IsoWithoutHot_MuonEnergy_phi[maxDep_],
      *h_IsoWithoutHot_MuonEnergy_muon_eta[maxDep_];
  TProfile *h_HotWithoutIso_MuonEnergy_eta[maxDep_], *h_HotWithoutIso_MuonEnergy_phi[maxDep_],
      *h_HotWithoutIso_MuonEnergy_muon_eta[maxDep_];
};

HBHEMuonOfflineAnalyzer::HBHEMuonOfflineAnalyzer(TChain *tree,
                                                 const char *outFileName,
                                                 const char *rcorFileName,
                                                 int flag,
                                                 int mode,
                                                 int maxDHB,
                                                 int maxDHE,
                                                 int cutMu,
                                                 float cutP,
                                                 int nevMax,
                                                 int over,
                                                 int runLo,
                                                 int runHi,
                                                 int etaMin,
                                                 int etaMax,
                                                 bool deb)
    : nCut_(cutMu), cutP_(cutP), nevMax_(nevMax), over_(over == 1), debug_(deb), cFactor_(false) {
  if ((nCut_ < 0) || (nCut_ > 2))
    nCut_ = 0;
  Init(tree, rcorFileName, flag, mode, maxDHB, maxDHE, runLo, runHi, etaMin, etaMax);

  //Now book histograms
  bookHistograms(outFileName);
}

HBHEMuonOfflineAnalyzer::HBHEMuonOfflineAnalyzer(const char *infile,
                                                 const char *outFileName,
                                                 const char *rcorFileName,
                                                 int flag,
                                                 int mode,
                                                 int maxDHB,
                                                 int maxDHE,
                                                 int cutMu,
                                                 float cutP,
                                                 int nevMax,
                                                 int over,
                                                 int runLo,
                                                 int runHi,
                                                 int etaMin,
                                                 int etaMax,
                                                 bool deb)
    : nCut_(cutMu), cutP_(cutP), nevMax_(nevMax), over_(over == 1), debug_(deb), cFactor_(false) {
  if ((nCut_ < 0) || (nCut_ > 2))
    nCut_ = 0;
  TChain *chain = new TChain("hcalHBHEMuon/TREE");
  if (!fillChain(chain, infile)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain, rcorFileName, flag, mode, maxDHB, maxDHE, runLo, runHi, etaMin, etaMax);

    //Now book histograms
    bookHistograms(outFileName);
  }
}

HBHEMuonOfflineAnalyzer::~HBHEMuonOfflineAnalyzer() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t HBHEMuonOfflineAnalyzer::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Int_t HBHEMuonOfflineAnalyzer::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t HBHEMuonOfflineAnalyzer::LoadTree(Long64_t entry) {
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

void HBHEMuonOfflineAnalyzer::Init(TChain *tree,
                                   const char *rcorFileName,
                                   int flag,
                                   int mode,
                                   int maxDHB,
                                   int maxDHE,
                                   int runLo,
                                   int runHi,
                                   int etaMin,
                                   int etaMax) {
  modeLHC_ = mode;
  maxDepthHB_ = maxDHB;
  maxDepthHE_ = maxDHE;
  maxDepth_ = (maxDepthHB_ > maxDepthHE_) ? maxDepthHB_ : maxDepthHE_;
  runLo_ = runLo;
  runHi_ = runHi;
  etaMin_ = (etaMin > 0) ? etaMin : 1;
  etaMax_ = (etaMax <= 29) ? etaMax : 29;
  if (etaMax_ <= etaMin_) {
    if (etaMax_ == 29)
      etaMin_ = etaMax_ - 1;
    else
      etaMax_ = etaMin_ + 1;
  }
  useCorrect_ = ((flag % 10) > 0);
  mergeDepth_ = (((flag / 10) % 10) > 0);
  if (std::string(rcorFileName) != "")
    cFactor_ = readCorr(rcorFileName);

  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer

  PF_Muon = 0;
  Global_Muon = 0;
  Tracker_muon = 0;
  pt_of_muon = 0;
  eta_of_muon = 0;
  phi_of_muon = 0;
  energy_of_muon = 0;
  p_of_muon = 0;
  muon_trkKink = 0;
  muon_chi2LocalPosition = 0;
  muon_segComp = 0;
  TrackerLayer = 0;
  NumPixelLayers = 0;
  InnerTrackPixelHits = 0;
  innerTrack = 0;
  chiTracker = 0;
  DxyTracker = 0;
  DzTracker = 0;
  innerTrackpt = 0;
  innerTracketa = 0;
  innerTrackphi = 0;
  tight_validFraction = 0;
  OuterTrack = 0;
  OuterTrackPt = 0;
  OuterTrackEta = 0;
  OuterTrackPhi = 0;
  OuterTrackHits = 0;
  OuterTrackRHits = 0;
  OuterTrackChi = 0;
  GlobalTrack = 0;
  GlobalTrckPt = 0;
  GlobalTrckEta = 0;
  GlobalTrckPhi = 0;
  Global_Muon_Hits = 0;
  MatchedStations = 0;
  GlobTrack_Chi = 0;
  Tight_LongitudinalImpactparameter = 0;
  Tight_TransImpactparameter = 0;
  IsolationR04 = 0;
  IsolationR03 = 0;
  ecal_3into3 = 0;
  hcal_3into3 = 0;
  tracker_3into3 = 0;
  matchedId = 0;
  hcal_cellHot = 0;
  ecal_3x3 = 0;
  hcal_1x1 = 0;
  ecal_detID = 0;
  hcal_detID = 0;
  ehcal_detID = 0;
  hcal_edepth1 = 0;
  hcal_activeL1 = 0;
  hcal_edepthHot1 = 0;
  hcal_activeHotL1 = 0;
  hcal_edepthCorrect1 = 0;
  hcal_edepthHotCorrect1 = 0;
  hcal_cdepthHot1 = 0;
  hcal_cdepthHotBG1 = 0;
  hcal_depthMatch1 = 0;
  hcal_depthMatchHot1 = 0;
  hcal_edepth2 = 0;
  hcal_activeL2 = 0;
  hcal_edepthHot2 = 0;
  hcal_activeHotL2 = 0;
  hcal_edepthCorrect2 = 0;
  hcal_edepthHotCorrect2 = 0;
  hcal_cdepthHot2 = 0;
  hcal_cdepthHotBG2 = 0;
  hcal_depthMatch2 = 0;
  hcal_depthMatchHot2 = 0;
  hcal_edepth3 = 0;
  hcal_activeL3 = 0;
  hcal_edepthHot3 = 0;
  hcal_activeHotL3 = 0;
  hcal_edepthCorrect3 = 0;
  hcal_edepthHotCorrect3 = 0;
  hcal_cdepthHot3 = 0;
  hcal_cdepthHotBG3 = 0;
  hcal_depthMatch3 = 0;
  hcal_depthMatchHot3 = 0;
  hcal_edepth4 = 0;
  hcal_activeL4 = 0;
  hcal_edepthHot4 = 0;
  hcal_activeHotL4 = 0;
  hcal_edepthCorrect4 = 0;
  hcal_edepthHotCorrect4 = 0;
  hcal_cdepthHot4 = 0;
  hcal_cdepthHotBG4 = 0;
  hcal_depthMatch4 = 0;
  hcal_depthMatchHot4 = 0;
  hcal_edepth5 = 0;
  hcal_activeL5 = 0;
  hcal_edepthHot5 = 0;
  hcal_activeHotL5 = 0;
  hcal_edepthCorrect5 = 0;
  hcal_edepthHotCorrect5 = 0;
  hcal_cdepthHot5 = 0;
  hcal_cdepthHotBG5 = 0;
  hcal_depthMatch5 = 0;
  hcal_depthMatchHot5 = 0;
  hcal_edepth6 = 0;
  hcal_activeL6 = 0;
  hcal_edepthHot6 = 0;
  hcal_activeHotL6 = 0;
  hcal_edepthCorrect6 = 0;
  hcal_edepthHotCorrect6 = 0;
  hcal_cdepthHot6 = 0;
  hcal_cdepthHotBG6 = 0;
  hcal_depthMatch6 = 0;
  hcal_depthMatchHot6 = 0;
  hcal_edepth7 = 0;
  hcal_activeL7 = 0;
  hcal_edepthHot7 = 0;
  hcal_activeHotL7 = 0;
  hcal_edepthCorrect7 = 0;
  hcal_edepthHotCorrect7 = 0;
  hcal_cdepthHot7 = 0;
  hcal_cdepthHotBG7 = 0;
  hcal_depthMatch7 = 0;
  hcal_depthMatchHot7 = 0;
  activeLength = 0;
  activeLengthHot = 0;
  hltresults = 0;
  all_triggers = 0;
  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("Event_No", &Event_No, &b_Event_No);
  fChain->SetBranchAddress("Run_No", &Run_No, &b_Run_No);
  fChain->SetBranchAddress("LumiNumber", &LumiNumber, &b_LumiNumber);
  fChain->SetBranchAddress("BXNumber", &BXNumber, &b_BXNumber);
  fChain->SetBranchAddress("GoodVertex", &GoodVertex, &b_GoodVertex);
  fChain->SetBranchAddress("PF_Muon", &PF_Muon, &b_PF_Muon);
  fChain->SetBranchAddress("Global_Muon", &Global_Muon, &b_Global_Muon);
  fChain->SetBranchAddress("Tracker_muon", &Tracker_muon, &b_Tracker_muon);
  fChain->SetBranchAddress("MuonIsTight", &MuonIsTight, &b_MuonIsTight);
  fChain->SetBranchAddress("MuonIsMedium", &MuonIsMedium, &b_MuonIsMedium);
  fChain->SetBranchAddress("pt_of_muon", &pt_of_muon, &b_pt_of_muon);
  fChain->SetBranchAddress("eta_of_muon", &eta_of_muon, &b_eta_of_muon);
  fChain->SetBranchAddress("phi_of_muon", &phi_of_muon, &b_phi_of_muon);
  fChain->SetBranchAddress("energy_of_muon", &energy_of_muon, &b_energy_of_muon);
  fChain->SetBranchAddress("p_of_muon", &p_of_muon, &b_p_of_muon);
  fChain->SetBranchAddress("muon_trkKink", &muon_trkKink, &b_muon_trkKink);
  fChain->SetBranchAddress("muon_chi2LocalPosition", &muon_chi2LocalPosition, &b_muon_chi2LocalPosition);
  fChain->SetBranchAddress("muon_segComp", &muon_segComp, &b_muon_segComp);
  fChain->SetBranchAddress("TrackerLayer", &TrackerLayer, &b_TrackerLayer);
  fChain->SetBranchAddress("NumPixelLayers", &NumPixelLayers, &b_NumPixelLayers);
  fChain->SetBranchAddress("InnerTrackPixelHits", &InnerTrackPixelHits, &b_InnerTrackPixelHits);
  fChain->SetBranchAddress("innerTrack", &innerTrack, &b_innerTrack);
  fChain->SetBranchAddress("chiTracker", &chiTracker, &b_chiTracker);
  fChain->SetBranchAddress("DxyTracker", &DxyTracker, &b_DxyTracker);
  fChain->SetBranchAddress("DzTracker", &DzTracker, &b_DzTracker);
  fChain->SetBranchAddress("innerTrackpt", &innerTrackpt, &b_innerTrackpt);
  fChain->SetBranchAddress("innerTracketa", &innerTracketa, &b_innerTracketa);
  fChain->SetBranchAddress("innerTrackphi", &innerTrackphi, &b_innerTrackphi);
  fChain->SetBranchAddress("tight_validFraction", &tight_validFraction, &b_tight_validFraction);
  fChain->SetBranchAddress("OuterTrack", &OuterTrack, &b_OuterTrack);
  fChain->SetBranchAddress("OuterTrackPt", &OuterTrackPt, &b_OuterTrackPt);
  fChain->SetBranchAddress("OuterTrackEta", &OuterTrackEta, &b_OuterTrackEta);
  fChain->SetBranchAddress("OuterTrackPhi", &OuterTrackPhi, &b_OuterTrackPhi);
  fChain->SetBranchAddress("OuterTrackChi", &OuterTrackChi, &b_OuterTrackChi);
  fChain->SetBranchAddress("OuterTrackHits", &OuterTrackHits, &b_OuterTrackHits);
  fChain->SetBranchAddress("OuterTrackRHits", &OuterTrackRHits, &b_OuterTrackRHits);
  fChain->SetBranchAddress("GlobalTrack", &GlobalTrack, &b_GlobalTrack);
  fChain->SetBranchAddress("GlobalTrckPt", &GlobalTrckPt, &b_GlobalTrckPt);
  fChain->SetBranchAddress("GlobalTrckEta", &GlobalTrckEta, &b_GlobalTrckEta);
  fChain->SetBranchAddress("GlobalTrckPhi", &GlobalTrckPhi, &b_GlobalTrckPhi);
  fChain->SetBranchAddress("Global_Muon_Hits", &Global_Muon_Hits, &b_Global_Muon_Hits);
  fChain->SetBranchAddress("MatchedStations", &MatchedStations, &b_MatchedStations);
  fChain->SetBranchAddress("GlobTrack_Chi", &GlobTrack_Chi, &b_GlobTrack_Chi);
  fChain->SetBranchAddress(
      "Tight_LongitudinalImpactparameter", &Tight_LongitudinalImpactparameter, &b_Tight_LongitudinalImpactparameter);
  fChain->SetBranchAddress("Tight_TransImpactparameter", &Tight_TransImpactparameter, &b_Tight_TransImpactparameter);
  fChain->SetBranchAddress("IsolationR04", &IsolationR04, &b_IsolationR04);
  fChain->SetBranchAddress("IsolationR03", &IsolationR03, &b_IsolationR03);
  fChain->SetBranchAddress("ecal_3into3", &ecal_3into3, &b_ecal_3into3);
  fChain->SetBranchAddress("hcal_3into3", &hcal_3into3, &b_hcal_3into3);
  fChain->SetBranchAddress("tracker_3into3", &tracker_3into3, &b_tracker_3into3);
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
  fChain->SetBranchAddress("hcal_edepthCorrect1", &hcal_edepthCorrect1, &b_hcal_edepthCorrect1);
  fChain->SetBranchAddress("hcal_edepthHotCorrect1", &hcal_edepthHotCorrect1, &b_hcal_edepthHotCorrect1);
  fChain->SetBranchAddress("hcal_cdepthHot1", &hcal_cdepthHot1, &b_hcal_cdepthHot1);
  fChain->SetBranchAddress("hcal_cdepthHotBG1", &hcal_cdepthHotBG1, &b_hcal_cdepthHotBG1);
  fChain->SetBranchAddress("hcal_depthMatch1", &hcal_depthMatch1, &b_hcal_depthMatch1);
  fChain->SetBranchAddress("hcal_depthMatchHot1", &hcal_depthMatchHot1, &b_hcal_depthMatchHot1);
  fChain->SetBranchAddress("hcal_edepth2", &hcal_edepth2, &b_hcal_edepth2);
  fChain->SetBranchAddress("hcal_activeL2", &hcal_activeL2, &b_hcal_activeL2);
  fChain->SetBranchAddress("hcal_edepthHot2", &hcal_edepthHot2, &b_hcal_edepthHot2);
  fChain->SetBranchAddress("hcal_activeHotL2", &hcal_activeHotL2, &b_hcal_activeHotL2);
  fChain->SetBranchAddress("hcal_edepthCorrect2", &hcal_edepthCorrect2, &b_hcal_edepthCorrect2);
  fChain->SetBranchAddress("hcal_edepthHotCorrect2", &hcal_edepthHotCorrect2, &b_hcal_edepthHotCorrect2);
  fChain->SetBranchAddress("hcal_cdepthHot2", &hcal_cdepthHot2, &b_hcal_cdepthHot2);
  fChain->SetBranchAddress("hcal_cdepthHotBG2", &hcal_cdepthHotBG2, &b_hcal_cdepthHotBG2);
  fChain->SetBranchAddress("hcal_depthMatch2", &hcal_depthMatch2, &b_hcal_depthMatch2);
  fChain->SetBranchAddress("hcal_depthMatchHot2", &hcal_depthMatchHot2, &b_hcal_depthMatchHot2);
  fChain->SetBranchAddress("hcal_edepth3", &hcal_edepth3, &b_hcal_edepth3);
  fChain->SetBranchAddress("hcal_activeL3", &hcal_activeL3, &b_hcal_activeL3);
  fChain->SetBranchAddress("hcal_edepthHot3", &hcal_edepthHot3, &b_hcal_edepthHot3);
  fChain->SetBranchAddress("hcal_activeHotL3", &hcal_activeHotL3, &b_hcal_activeHotL3);
  fChain->SetBranchAddress("hcal_edepthCorrect3", &hcal_edepthCorrect3, &b_hcal_edepthCorrect3);
  fChain->SetBranchAddress("hcal_edepthHotCorrect3", &hcal_edepthHotCorrect3, &b_hcal_edepthHotCorrect3);
  fChain->SetBranchAddress("hcal_cdepthHot3", &hcal_cdepthHot3, &b_hcal_cdepthHot3);
  fChain->SetBranchAddress("hcal_cdepthHotBG3", &hcal_cdepthHotBG3, &b_hcal_cdepthHotBG3);
  fChain->SetBranchAddress("hcal_depthMatch3", &hcal_depthMatch3, &b_hcal_depthMatch3);
  fChain->SetBranchAddress("hcal_depthMatchHot3", &hcal_depthMatchHot3, &b_hcal_depthMatchHot3);
  fChain->SetBranchAddress("hcal_edepth4", &hcal_edepth4, &b_hcal_edepth4);
  fChain->SetBranchAddress("hcal_activeL4", &hcal_activeL4, &b_hcal_activeL4);
  fChain->SetBranchAddress("hcal_edepthHot4", &hcal_edepthHot4, &b_hcal_edepthHot4);
  fChain->SetBranchAddress("hcal_activeHotL4", &hcal_activeHotL4, &b_hcal_activeHotL4);
  fChain->SetBranchAddress("hcal_edepthCorrect4", &hcal_edepthCorrect4, &b_hcal_edepthCorrect4);
  fChain->SetBranchAddress("hcal_edepthHotCorrect4", &hcal_edepthHotCorrect4, &b_hcal_edepthHotCorrect4);
  fChain->SetBranchAddress("hcal_cdepthHot4", &hcal_cdepthHot4, &b_hcal_cdepthHot4);
  fChain->SetBranchAddress("hcal_cdepthHotBG4", &hcal_cdepthHotBG4, &b_hcal_cdepthHotBG4);
  fChain->SetBranchAddress("hcal_depthMatch4", &hcal_depthMatch4, &b_hcal_depthMatch4);
  fChain->SetBranchAddress("hcal_depthMatchHot4", &hcal_depthMatchHot4, &b_hcal_depthMatchHot4);
  fChain->SetBranchAddress("hcal_edepth5", &hcal_edepth5, &b_hcal_edepth5);
  fChain->SetBranchAddress("hcal_activeL5", &hcal_activeL5, &b_hcal_activeL5);
  fChain->SetBranchAddress("hcal_edepthHot5", &hcal_edepthHot5, &b_hcal_edepthHot5);
  fChain->SetBranchAddress("hcal_activeHotL5", &hcal_activeHotL5, &b_hcal_activeHotL5);
  fChain->SetBranchAddress("hcal_edepthCorrect5", &hcal_edepthCorrect5, &b_hcal_edepthCorrect5);
  fChain->SetBranchAddress("hcal_edepthHotCorrect5", &hcal_edepthHotCorrect5, &b_hcal_edepthHotCorrect5);
  fChain->SetBranchAddress("hcal_cdepthHot5", &hcal_cdepthHot5, &b_hcal_cdepthHot5);
  fChain->SetBranchAddress("hcal_cdepthHotBG5", &hcal_cdepthHotBG5, &b_hcal_cdepthHotBG5);
  fChain->SetBranchAddress("hcal_depthMatch5", &hcal_depthMatch5, &b_hcal_depthMatch5);
  fChain->SetBranchAddress("hcal_depthMatchHot5", &hcal_depthMatchHot5, &b_hcal_depthMatchHot5);
  fChain->SetBranchAddress("hcal_edepth6", &hcal_edepth6, &b_hcal_edepth6);
  fChain->SetBranchAddress("hcal_activeL6", &hcal_activeL6, &b_hcal_activeL6);
  fChain->SetBranchAddress("hcal_edepthHot6", &hcal_edepthHot6, &b_hcal_edepthHot6);
  fChain->SetBranchAddress("hcal_activeHotL6", &hcal_activeHotL6, &b_hcal_activeHotL6);
  fChain->SetBranchAddress("hcal_edepthCorrect6", &hcal_edepthCorrect6, &b_hcal_edepthCorrect6);
  fChain->SetBranchAddress("hcal_edepthHotCorrect6", &hcal_edepthHotCorrect6, &b_hcal_edepthHotCorrect6);
  fChain->SetBranchAddress("hcal_cdepthHot6", &hcal_cdepthHot6, &b_hcal_cdepthHot6);
  fChain->SetBranchAddress("hcal_cdepthHotBG6", &hcal_cdepthHotBG6, &b_hcal_cdepthHotBG6);
  fChain->SetBranchAddress("hcal_depthMatch6", &hcal_depthMatch6, &b_hcal_depthMatch6);
  fChain->SetBranchAddress("hcal_depthMatchHot6", &hcal_depthMatchHot6, &b_hcal_depthMatchHot6);
  fChain->SetBranchAddress("hcal_edepth7", &hcal_edepth7, &b_hcal_edepth7);
  fChain->SetBranchAddress("hcal_activeL7", &hcal_activeL7, &b_hcal_activeL7);
  fChain->SetBranchAddress("hcal_edepthHot7", &hcal_edepthHot7, &b_hcal_edepthHot7);
  fChain->SetBranchAddress("hcal_activeHotL7", &hcal_activeHotL7, &b_hcal_activeHotL7);
  fChain->SetBranchAddress("hcal_edepthCorrect7", &hcal_edepthCorrect7, &b_hcal_edepthCorrect7);
  fChain->SetBranchAddress("hcal_edepthHotCorrect7", &hcal_edepthHotCorrect7, &b_hcal_edepthHotCorrect7);
  fChain->SetBranchAddress("hcal_cdepthHot7", &hcal_cdepthHot7, &b_hcal_cdepthHot7);
  fChain->SetBranchAddress("hcal_cdepthHotBG7", &hcal_cdepthHotBG7, &b_hcal_cdepthHotBG7);
  fChain->SetBranchAddress("hcal_depthMatch7", &hcal_depthMatch7, &b_hcal_depthMatch7);
  fChain->SetBranchAddress("hcal_depthMatchHot7", &hcal_depthMatchHot7, &b_hcal_depthMatchHot7);
  fChain->SetBranchAddress("activeLength", &activeLength, &b_activeLength);
  fChain->SetBranchAddress("activeLengthHot", &activeLengthHot, &b_activeLengthHot);
  fChain->SetBranchAddress("hltresults", &hltresults, &b_hltresults);
  fChain->SetBranchAddress("all_triggers", &all_triggers, &b_all_triggers);

  Notify();
}

void HBHEMuonOfflineAnalyzer::Loop() {
  //declarations
  if (fChain == 0)
    return;

  Long64_t nentries = fChain->GetEntriesFast();
  if (debug_)
    std::cout << "nevent = " << nentries << std::endl;
  if (nevMax_ > 0)
    nentries = nevMax_;

  Long64_t nbytes = 0, nb = 0, nsel1 = 0, nsel2 = 0;
  Long64_t nstep1 = 0, nstep2 = 0, nstep3 = 0, nstep4 = 0;

  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if ((int)(Run_No) < runLo_ || (int)(Run_No) > runHi_)
      continue;
    ++nstep1;
    if (debug_)
      std::cout << "Run " << Run_No << " Event " << Event_No << " Muon pt " << pt_of_muon << std::endl;

    bool loose(false), soft(false), tight(false), pcut(false), ptcut(false);
    t_ene.clear();
    t_enec.clear();
    t_charge.clear();
    t_actln.clear();
    t_depth.clear();

    if (debug_)
      std::cout << "ecal_det_id " << ecal_detID << std::endl;

    int typeEcal, etaXEcal, phiYEcal, zsideEcal, planeEcal, stripEcal;
    etaPhiEcal(ecal_detID, typeEcal, zsideEcal, etaXEcal, phiYEcal, planeEcal, stripEcal);
    double etaEcal = (etaXEcal - 0.5) * zsideEcal;
    double phiEcal = phiYEcal - 0.5;

    if (debug_)
      std::cout << "hcal_det_id " << std::hex << hcal_detID << std::dec;

    int etaHcal, phiHcal, depthHcal;
    etaPhiHcal(hcal_detID, etaHcal, phiHcal, depthHcal);

    int eta = (etaHcal > 0) ? etaHcal - 1 : -etaHcal - 1;
    double etaXHcal = (etaHcal > 0) ? etaHcal - 0.5 : etaHcal + 0.5;
    int nDepth = nDepthBins(eta + 1, phiHcal);
    int nPhi = nPhiBins(eta + 1);
    int PHI = (nPhi > 36) ? (phiHcal - 1) : (phiHcal - 1) / 2;
    double phiYHcal = (phiHcal - 0.5);
    t_ieta = etaHcal;
    t_iphi = PHI;
    t_p = p_of_muon;
    t_ediff = hcal_3into3 - hcal_1x1;
    t_nvtx = GoodVertex;
    if (p_of_muon > cutP_)
      pcut = true;
    if (pt_of_muon > cutP_)
      ptcut = true;
    if (looseMuon())
      loose = true;
    if (softMuon())
      soft = true;
    if (tightMuon())
      tight = true;

    if (debug_)
      std::cout << " etaHcal " << etaHcal << ":" << etaXHcal << " phiHcal " << phiHcal << ":" << phiYHcal << ":" << PHI
                << " Depth " << nDepth << " Muon Pt " << pt_of_muon << " Isol " << IsolationR04 << std::endl;

    int cut(nCut_);
    bool select(false);
    if (cut == 0)
      select = tightMuon();
    else if (cut == 1)
      select = softMuon();
    else
      select = looseMuon();

    if (select)
      ++nstep2;
    if (select && ((eta + 1) >= etaMin_) && ((eta + 1) <= etaMax_)) {
      ++nstep3;
      h_Pt_Muon->Fill(pt_of_muon);
      h_Eta_Muon->Fill(eta_of_muon);
      h_Phi_Muon->Fill(phi_of_muon);
      h_PF_Muon->Fill(PF_Muon);
      h_GlobTrack_Chi->Fill(GlobTrack_Chi);
      h_Global_Muon_Hits->Fill(Global_Muon_Hits);
      h_MatchedStations->Fill(MatchedStations);
      h_Tight_TransImpactparameter->Fill(Tight_TransImpactparameter);
      h_Tight_LongitudinalImpactparameter->Fill(Tight_LongitudinalImpactparameter);
      h_InnerTrackPixelHits->Fill(InnerTrackPixelHits);
      h_TrackerLayer->Fill(TrackerLayer);
      h_IsolationR04->Fill(IsolationR04);
      h_Global_Muon->Fill(Global_Muon);

      h_TransImpactParameter->Fill(Tight_TransImpactparameter);
      h_LongImpactParameter->Fill(Tight_LongitudinalImpactparameter);

      //in Phi Bins
      if (((phi_of_muon) >= -1.5) || ((phi_of_muon) <= 0.5)) {
        h_TransImpactParameterBin1->Fill(Tight_TransImpactparameter);
        h_LongImpactParameterBin1->Fill(Tight_LongitudinalImpactparameter);
        h_2D_Bin1->Fill(Tight_TransImpactparameter, Tight_LongitudinalImpactparameter);
      }

      if ((phi_of_muon > 0.5) || (phi_of_muon < -1.5)) {
        h_TransImpactParameterBin2->Fill(Tight_TransImpactparameter);
        h_LongImpactParameterBin2->Fill(Tight_LongitudinalImpactparameter);
        h_2D_Bin2->Fill(Tight_TransImpactparameter, Tight_LongitudinalImpactparameter);
      }

      h_ecal_energy->Fill(ecal_3into3);
      h_3x3_ecal->Fill(ecal_3x3);
      h_Eta_ecal->Fill(eta_of_muon, ecal_3x3);
      h_Phi_ecal->Fill(phi_of_muon, ecal_3x3);
      h_MuonHittingEcal->Fill(typeEcal);
      if (typeEcal == 1) {
        h_EtaX_ecal->Fill(etaEcal, ecal_3x3);
        h_PhiY_ecal->Fill(phiEcal, ecal_3x3);
      }

      h_hcal_energy->Fill(hcal_3into3);
      h_1x1_hcal->Fill(hcal_1x1);
      h_EtaX_hcal->Fill(etaXHcal, hcal_1x1);
      h_PhiY_hcal->Fill(phiYHcal, hcal_1x1);
      h_HotCell->Fill(hcal_cellHot);
      if (mergeDepth_) {
        double en1(0), en2(0), actLTot(0), chargeS(0), chargeBG(0);
        double enh(0), enc(0);
        for (int dep = 0; dep < nDepth; ++dep) {
          double enb(0), enu(0), eh0(0), ec0(0), chgS(0), chgB(0), actL(0);
          getEnergy(dep, enb, enu, eh0, ec0, chgS, chgB, actL);
          en1 += ((useCorrect_) ? enu : enb);
          en2 += ((useCorrect_) ? ec0 : eh0);
          enh += (eh0);
          enc += (ec0);
          actLTot += (actL);
          chargeS += (chgS);
          chargeBG += (chgB);
        }
        int ind = (etaHcal > 0) ? indxEta[eta][0][PHI] : 1 + indxEta[eta][0][PHI];
        if (debug_)  // || eta==15 || eta==17)
          std::cout << "Matched Id " << matchedId << " Hot " << hcal_cellHot << " eta " << etaHcal << ":" << eta
                    << " phi " << phiHcal << ":" << PHI << " Index " << ind << " E " << en1 << ":" << en2 << ":" << enh
                    << ":" << enc << " L " << actLTot << " Charge " << chargeS << ":" << chargeBG << std::endl;
        if (!(matchedId) && !(over_))
          continue;
        if ((hcal_cellHot == 1) || over_) {
          if (actLTot > 0) {
            h_Hot_MuonEnergy_hcal_HotCell[ind]->Fill(en2);
            h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ind]->Fill(en2 / actLTot);
            h_active_length_Fill[ind]->Fill(actLTot);
            h_p_muon_ineta[ind]->Fill(p_of_muon);
            h_charge_signal[ind]->Fill(chargeS);
            h_charge_bg[ind]->Fill(chargeBG);

            t_ene.push_back(enh);
            t_enec.push_back(enc);
            t_charge.push_back(chargeS);
            t_actln.push_back(actLTot);
            t_depth.push_back(0);

            outtree_->Fill();
            ++nsel1;
          }
        }
      } else {
        bool fillTree(false);
        ++nstep4;
        for (int dep = 0; dep < nDepth; ++dep) {
          if (debug_)
            std::cout << "dep:" << dep << std::endl;

          double actL(0), chargeS(-9999), chargeBG(-9999);
          double enh(-9999), enc(-9999), enb(0), enu(0);
          bool ok1 = getEnergy(dep, enb, enu, enh, enc, chargeS, chargeBG, actL);
          double en1 = ((useCorrect_) ? enu : enb);
          double en2 = ((useCorrect_) ? enc : enh);
          if (debug_)
            std::cout << "Hello in " << dep + 1 << " " << en1 << ":" << en2 << ":" << actL << std::endl;

          bool ok2 = ok1;

          if (debug_)
            std::cout << "Before Index " << ok1 << ":" << ok2 << std::endl;

          int ind = (etaHcal > 0) ? indxEta[eta][dep][PHI] : 1 + indxEta[eta][dep][PHI];
          if (debug_)  // || eta==15 || eta==17)
            std::cout << "Matched Id " << matchedId << " Hot " << hcal_cellHot << " eta " << etaHcal << ":" << eta
                      << " phi " << phiHcal << ":" << PHI << " depth " << dep << " Index " << ind << " E " << en1 << ":"
                      << en2 << ":" << enh << ":" << enc << " L " << actL << " Charge " << chargeS << ":" << chargeBG
                      << std::endl;
          if (debug_)
            std::cout << "matchedId " << matchedId << " Over " << over_ << " OK " << ok1 << ":" << ok2 << " cellHot "
                      << (hcal_cellHot == 1) << std::endl;
          if (!(matchedId) && !(over_))
            continue;
          if (ok1 || over_) {
            if (debug_)
              std::cout << "enter ok1" << std::endl;

            if ((hcal_cellHot == 1) || (over_)) {
              if (actL > 0) {
                h_Hot_MuonEnergy_hcal_HotCell[ind]->Fill(en2);
                h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ind]->Fill(en2 / actL);
                h_active_length_Fill[ind]->Fill(actL);
                h_p_muon_ineta[ind]->Fill(p_of_muon);
                h_charge_signal[ind]->Fill(chargeS);
                h_charge_bg[ind]->Fill(chargeBG);
                t_ene.push_back(enh);
                t_enec.push_back(enc);
                t_charge.push_back(chargeS);
                t_actln.push_back(actL);
                // added depth vector AmanKalsi
                t_depth.push_back(dep);
                fillTree = true;
              } else {
                t_ene.push_back(-999.0);
                t_enec.push_back(-999.0);
                t_charge.push_back(-999.0);
                t_actln.push_back(-999.0);
                t_depth.push_back(-999.0);
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
        if (fillTree) {
          outtree_->Fill();
          ++nsel2;
        }
      }
    }
    int evtype(0);
    if (pcut)
      evtype += 1;
    if (ptcut)
      evtype += 2;
    if (loose)
      evtype += 4;
    if (soft)
      evtype += 8;
    if (tight)
      evtype += 16;
    h_evtype->Fill(evtype);
  }
  std::cout << "Number of events in the output root tree: " << nsel1 << ":" << nsel2 << ":" << nstep1 << ":" << nstep2
            << ":" << nstep3 << ":" << nstep4 << std::endl;
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
  if (!fChain)
    return;
  fChain->Show(entry);
}

bool HBHEMuonOfflineAnalyzer::fillChain(TChain *chain, const char *inputFileList) {
  std::string fname(inputFileList);
  if (fname.substr(fname.size() - 5, 5) == ".root") {
    chain->Add(fname.c_str());
  } else {
    std::ifstream infile(inputFileList);
    if (!infile.is_open()) {
      std::cout << "** ERROR: Can't open '" << inputFileList << "' for input" << std::endl;
      return false;
    }
    while (1) {
      infile >> fname;
      if (!infile.good())
        break;
      chain->Add(fname.c_str());
    }
    infile.close();
  }
  std::cout << "No. of Entries in this tree : " << chain->GetEntries() << std::endl;
  return true;
}

bool HBHEMuonOfflineAnalyzer::readCorr(const char *infile) {
  std::ifstream fInput(infile);
  unsigned int ncorr(0), all(0), good(0);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::string bufferString(buffer);
      if (bufferString.substr(0, 5) == "#IOVs") {
        std::vector<std::string> items = splitString(bufferString.substr(6));
        ncorr = items.size() - 1;
        for (unsigned int n = 0; n < ncorr; ++n) {
          int run = std::atoi(items[n].c_str());
          runlow_.push_back(run);
        }
        std::cout << ncorr << ":" << runlow_.size() << " Run ranges" << std::endl;
        for (unsigned int n = 0; n < runlow_.size(); ++n)
          std::cout << " [" << n << "] " << runlow_[n];
        std::cout << std::endl;
      } else if (buffer[0] == '#') {
        continue;  //ignore other comments
      } else {
        std::vector<std::string> items = splitString(bufferString);
        if (items.size() != ncorr + 3) {
          std::cout << "Ignore  line: " << buffer << std::endl;
        } else {
          ++good;
          int ieta = std::atoi(items[0].c_str());
          int iphi = std::atoi(items[1].c_str());
          int depth = std::atoi(items[2].c_str());
          unsigned int id = getDetIdHBHE(ieta, iphi, depth);
          for (unsigned int n = 0; n < ncorr; ++n) {
            float corrf = std::atof(items[n + 3].c_str());
            if (n < nmax_)
              corrFac_[n][id] = corrf;
          }
          if (debug_) {
            std::cout << "ID " << std::hex << id << std::dec << ":" << id << " (eta " << ieta << " phi " << iphi
                      << " depth " << depth << ")";
            for (unsigned int n = 0; n < ncorr; ++n)
              std::cout << " " << corrFac_[n][id];
            std::cout << std::endl;
          }
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records" << std::endl;
  }
  return (good > 0);
}

void HBHEMuonOfflineAnalyzer::bookHistograms(const char *fname) {
  std::cout << "BookHistograms" << std::endl;
  output_file = TFile::Open(fname, "RECREATE");
  output_file->cd();
  outtree_ = new TTree("Lep_Tree", "Lep_Tree");
  outtree_->Branch("t_ieta", &t_ieta);
  outtree_->Branch("t_iphi", &t_iphi);
  outtree_->Branch("t_nvtx", &t_nvtx);
  outtree_->Branch("t_p", &t_p);
  outtree_->Branch("t_ediff", &t_ediff);
  outtree_->Branch("t_ene", &t_ene);
  outtree_->Branch("t_enec", &t_enec);
  outtree_->Branch("t_charge", &t_charge);
  outtree_->Branch("t_actln", &t_actln);
  outtree_->Branch("t_depth", &t_depth);

  std::string type[] = {"tight", "soft", "loose"};
  char name[128], title[500];

  nHist = 0;
  for (int eta = etaMin_; eta <= etaMax_; ++eta) {
    int nDepth = nDepthBins(eta, -1);
    int nPhi = nPhiBins(eta);
    for (int depth = 0; depth < nDepth; depth++) {
      for (int PHI = 0; PHI < nPhi; ++PHI) {
        indxEta[eta - 1][depth][PHI] = nHist;
        nHist += 2;
      }
    }
  }
  if (nHist >= maxHist_) {
    std::cout << "Problem here " << nHist << ":" << maxHist_ << std::endl;
  }
  //	TDirectory *d_output_file[nCut_][29];
  //output_file->cd();
  h_evtype = new TH1D("EvType", "Event Type", 100, 0, 100);
  int i(nCut_);
  sprintf(name, "h_Pt_Muon_%s", type[i].c_str());
  sprintf(title, "p_{T} of %s muons (GeV)", type[i].c_str());
  h_Pt_Muon = new TH1D(name, title, 100, 0, 200);

  sprintf(name, "h_Eta_Muon_%s", type[i].c_str());
  sprintf(title, "#eta of %s muons", type[i].c_str());
  h_Eta_Muon = new TH1D(name, title, 50, -2.5, 2.5);

  sprintf(name, "h_Phi_Muon_%s", type[i].c_str());
  sprintf(title, "#phi of %s muons", type[i].c_str());
  h_Phi_Muon = new TH1D(name, title, 100, -3.1415926, 3.1415926);

  sprintf(name, "h_P_Muon_%s", type[i].c_str());
  sprintf(title, "p of %s muons (GeV)", type[i].c_str());
  h_P_Muon = new TH1D(name, title, 100, 0, 200);

  sprintf(name, "h_PF_Muon_%s", type[i].c_str());
  sprintf(title, "PF %s muons (GeV)", type[i].c_str());
  h_PF_Muon = new TH1D(name, title, 2, 0, 2);

  sprintf(name, "h_Global_Muon_Chi2_%s", type[i].c_str());
  sprintf(title, "Chi2 Global %s muons (GeV)", type[i].c_str());
  h_GlobTrack_Chi = new TH1D(name, title, 15, 0, 15);

  sprintf(name, "h_Global_Muon_Hits_%s", type[i].c_str());
  sprintf(title, "Global Hits %s muons (GeV)", type[i].c_str());
  h_Global_Muon_Hits = new TH1D(name, title, 10, 0, 10);

  sprintf(name, "h_Matched_Stations_%s", type[i].c_str());
  sprintf(title, "Matched Stations %s muons (GeV)", type[i].c_str());
  h_MatchedStations = new TH1D(name, title, 10, 0, 10);

  sprintf(name, "h_Transverse_ImpactParameter_%s", type[i].c_str());
  sprintf(title, "Transverse_ImpactParameter of %s muons (GeV)", type[i].c_str());
  h_Tight_TransImpactparameter = new TH1D(name, title, 50, 0, 10);

  sprintf(name, "h_Longitudinal_ImpactParameter_%s", type[i].c_str());
  sprintf(title, "Longitudinal_ImpactParameter of %s muons (GeV)", type[i].c_str());
  h_Tight_LongitudinalImpactparameter = new TH1D(name, title, 20, 0, 10);

  sprintf(name, "h_InnerTrack_PixelHits_%s", type[i].c_str());
  sprintf(title, "InnerTrack_PixelHits of %s muons (GeV)", type[i].c_str());
  h_InnerTrackPixelHits = new TH1D(name, title, 20, 0, 20);

  sprintf(name, "h_TrackLayers_%s", type[i].c_str());
  sprintf(title, "No. of Tracker Layers of %s muons (GeV)", type[i].c_str());
  h_TrackerLayer = new TH1D(name, title, 20, 0, 20);

  sprintf(name, "h_IsolationR04_%s", type[i].c_str());
  sprintf(title, "IsolationR04 %s muons (GeV)", type[i].c_str());
  h_IsolationR04 = new TH1D(name, title, 45, 0, 5);

  sprintf(name, "h_Global_Muon_%s", type[i].c_str());
  sprintf(title, "Global %s muons (GeV)", type[i].c_str());
  h_Global_Muon = new TH1D(name, title, 2, 0, 2);

  sprintf(name, "h_TransImpactParameter_%s", type[i].c_str());
  sprintf(title, "TransImpactParameter of %s muons (GeV)", type[i].c_str());
  h_TransImpactParameter = new TH1D(name, title, 100, 0, 0.5);

  sprintf(name, "h_TransImpactParameterBin1_%s", type[i].c_str());
  sprintf(title, "TransImpactParameter of %s muons (GeV) in -1.5 <= #phi <= 0.5", type[i].c_str());
  h_TransImpactParameterBin1 = new TH1D(name, title, 100, 0, 0.5);

  sprintf(name, "h_TransImpactParameterBin2_%s", type[i].c_str());
  sprintf(title, "TransImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
  h_TransImpactParameterBin2 = new TH1D(name, title, 100, 0, 0.5);

  sprintf(name, "h_LongImpactParameter_%s", type[i].c_str());
  sprintf(title, "LongImpactParameter of %s muons (GeV)", type[i].c_str());
  h_LongImpactParameter = new TH1D(name, title, 100, 0, 30);

  sprintf(name, "h_LongImpactParameterBin1_%s", type[i].c_str());
  sprintf(title, "LongImpactParameter of %s muons (GeV) in -1.5 <= #phi <= 0.5", type[i].c_str());
  h_LongImpactParameterBin1 = new TH1D(name, title, 100, 0, 30);

  sprintf(name, "h_LongImpactParameterBin2_%s", type[i].c_str());
  sprintf(title, "LongImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
  h_LongImpactParameterBin2 = new TH1D(name, title, 100, 0, 30);

  sprintf(name, "h_2D_Bin1_%s", type[i].c_str());
  sprintf(title, "Trans/Long ImpactParameter of %s muons (GeV) in -1.5 <= #phi< 0.5 ", type[i].c_str());
  h_2D_Bin1 = new TH2D(name, title, 100, 0, 0.5, 100, 0, 30);

  sprintf(name, "h_2D_Bin2_%s", type[i].c_str());
  sprintf(title, "Trans/Long ImpactParameter of %s muons (GeV) in #phi> 0.5 and #phi< -1.5 ", type[i].c_str());
  h_2D_Bin2 = new TH2D(name, title, 100, 0, 0.5, 100, 0, 30);

  sprintf(name, "h_ecal_energy_%s", type[i].c_str());
  sprintf(title, "ECAL energy for %s muons", type[i].c_str());
  h_ecal_energy = new TH1D(name, title, 1000, -10.0, 90.0);

  sprintf(name, "h_hcal_energy_%s", type[i].c_str());
  sprintf(title, "HCAL energy for %s muons", type[i].c_str());
  h_hcal_energy = new TH1D(name, title, 500, -10.0, 90.0);

  sprintf(name, "h_3x3_ecal_%s", type[i].c_str());
  sprintf(title, "ECAL energy in 3x3 for %s muons", type[i].c_str());
  h_3x3_ecal = new TH1D(name, title, 1000, -10.0, 90.0);

  sprintf(name, "h_1x1_hcal_%s", type[i].c_str());
  sprintf(title, "HCAL energy in 1x1 for %s muons", type[i].c_str());
  h_1x1_hcal = new TH1D(name, title, 500, -10.0, 90.0);

  sprintf(name, "h_EtaX_hcal_%s", type[i].c_str());
  sprintf(title, "HCAL energy as a function of i#eta for %s muons", type[i].c_str());
  h_EtaX_hcal = new TProfile(name, title, 60, -30.0, 30.0);

  sprintf(name, "h_PhiY_hcal_%s", type[i].c_str());
  sprintf(title, "HCAL energy as a function of i#phi for %s muons", type[i].c_str());
  h_PhiY_hcal = new TProfile(name, title, 72, 0, 72);

  sprintf(name, "h_EtaX_ecal_%s", type[i].c_str());
  sprintf(title, "EB energy as a function of i#eta for %s muons", type[i].c_str());
  h_EtaX_ecal = new TProfile(name, title, 170, -85.0, 85.0);

  sprintf(name, "h_PhiY_ecal_%s", type[i].c_str());
  sprintf(title, "EB energy as a function of i#phi for %s muons", type[i].c_str());
  h_PhiY_ecal = new TProfile(name, title, 360, 0, 360);

  sprintf(name, "h_Eta_ecal_%s", type[i].c_str());
  sprintf(title, "ECAL energy as a function of #eta for %s muons", type[i].c_str());
  h_Eta_ecal = new TProfile(name, title, 100, -2.5, 2.5);

  sprintf(name, "h_Phi_ecal_%s", type[i].c_str());
  sprintf(title, "ECAL energy as a function of #phi for %s muons", type[i].c_str());
  h_Phi_ecal = new TProfile(name, title, 100, -3.1415926, 3.1415926);

  sprintf(name, "h_MuonHittingEcal_%s", type[i].c_str());
  sprintf(title, "%s muons hitting ECAL", type[i].c_str());
  h_MuonHittingEcal = new TH1D(name, title, 100, 0, 5.0);

  sprintf(name, "h_HotCell_%s", type[i].c_str());
  sprintf(title, "Hot cell for %s muons", type[i].c_str());
  h_HotCell = new TH1D(name, title, 100, 0, 2);

  //		output_file->cd();
  for (int eta = etaMin_; eta <= etaMax_; ++eta) {
    int nDepth = nDepthBins(eta, -1);
    int nPhi = nPhiBins(eta);
    //sprintf(name, "Dir_muon_type_%s_ieta%d",type[i].c_str(), eta);
    //d_output_file[i][eta]= output_file->mkdir(name);
    //output_file->cd(name);
    //d_output_file[i][eta]->cd();
    for (int depth = 0; depth < nDepth; ++depth) {
      for (int PHI = 0; PHI < nPhi; ++PHI) {
        int PHI0 = (nPhi == 72) ? PHI + 1 : 2 * PHI + 1;
        int ih = indxEta[eta - 1][depth][PHI];
        if (debug_)
          std::cout << "eta:" << eta << " depth:" << depth << " PHI:" << PHI << ":" << PHI0 << " ih:" << ih
                    << std::endl;

        sprintf(name, "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell", eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title,
                "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi = %d) for extrapolated %s muons (Hot Cell)",
                eta,
                (depth + 1),
                PHI0,
                type[i].c_str());
        h_Hot_MuonEnergy_hcal_HotCell[ih] = new TH1D(name, title, 4000, 0.0, 10.0);
        h_Hot_MuonEnergy_hcal_HotCell[ih]->Sumw2();

        sprintf(
            name, "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell_ByActiveLength", eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title,
                "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi = %d) for extrapolated %s muons (Hot Cell) "
                "divided by Active Length",
                eta,
                (depth + 1),
                PHI0,
                type[i].c_str());
        h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ih] = new TH1D(name, title, 4000, 0.0, 10.0);
        h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ih]->Sumw2();

        sprintf(name, "h_active_length_Fill_%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title, "active_length%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        h_active_length_Fill[ih] = new TH1D(name, title, 20, 0, 20);
        h_active_length_Fill[ih]->Sumw2();

        sprintf(name, "h_p_muon_in_%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title, "p_muon_in%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        h_p_muon_ineta[ih] = new TH1D(name, title, 500, 0, 500);
        h_p_muon_ineta[ih]->Sumw2();

        sprintf(name, "h_charge_signal_in_%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(name, "charge_signal_in_%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        h_charge_signal[ih] = new TH1D(name, title, 500, 0, 500);
        h_charge_signal[ih]->Sumw2();

        sprintf(name, "h_charge_bg_in_%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(name, "charge_bg_in_%d_%d_%d_%s", eta, (depth + 1), PHI0, type[i].c_str());
        h_charge_bg[ih] = new TH1D(name, title, 500, 0, 500);
        h_charge_bg[ih]->Sumw2();

        ih++;

        sprintf(name, "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell", -eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title,
                "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi = %d) for extrapolated %s muons (Hot Cell)",
                -eta,
                (depth + 1),
                PHI0,
                type[i].c_str());
        h_Hot_MuonEnergy_hcal_HotCell[ih] = new TH1D(name, title, 4000, 0.0, 10.0);
        h_Hot_MuonEnergy_hcal_HotCell[ih]->Sumw2();

        sprintf(
            name, "h_Hot_MuonEnergy_hc_%d_%d_%d_%s_HotCell_ByActiveLength", -eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title,
                "HCAL energy in hot tower (i#eta=%d, depth=%d, i#phi=%d) for extrapolated %s muons (Hot Cell) "
                "divided by Active Length",
                -eta,
                (depth + 1),
                PHI0,
                type[i].c_str());
        h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ih] = new TH1D(name, title, 4000, 0.0, 10.0);
        h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ih]->Sumw2();

        sprintf(name, "h_active_length_Fill_%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title, "active_length%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        h_active_length_Fill[ih] = new TH1D(name, title, 20, 0, 20);
        h_active_length_Fill[ih]->Sumw2();

        sprintf(name, "h_p_muon_in_%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(title, "p_muon_in%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        h_p_muon_ineta[ih] = new TH1D(name, title, 500, 0, 500);
        h_p_muon_ineta[ih]->Sumw2();

        sprintf(name, "h_charge_signal_in_%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(name, "charge_signal_in_%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        h_charge_signal[ih] = new TH1D(name, title, 500, 0, 500);
        h_charge_signal[ih]->Sumw2();

        sprintf(name, "h_charge_bg_in_%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        sprintf(name, "charge_bg_in_%d_%d_%d_%s", -eta, (depth + 1), PHI0, type[i].c_str());
        h_charge_bg[ih] = new TH1D(name, title, 500, 0, 500);
        h_charge_bg[ih]->Sumw2();
      }
    }
    //output_file->cd();
  }
  //output_file->cd();
}

bool HBHEMuonOfflineAnalyzer::getEnergy(
    int dep, double &enb, double &enu, double &enh, double &enc, double &chgS, double &chgB, double &actL) {
  double cfac(1.0);
  bool flag(true);
  if (cFactor_) {
    int ieta = hcal_ieta;
    int iphi = hcal_iphi;
    unsigned int detId = getDetIdHBHE(ieta, iphi, dep + 1);
    cfac = getCorr(Run_No, detId);
  }
  if (dep == 0) {
    enb = cfac * hcal_edepth1;
    enu = cfac * hcal_edepthCorrect1;
    enh = cfac * hcal_edepthHot1;
    enc = cfac * hcal_edepthHotCorrect1;
    chgS = hcal_cdepthHot1;
    actL = hcal_activeHotL1;
    chgB = hcal_cdepthHotBG1;
  } else if (dep == 1) {
    enb = cfac * hcal_edepth2;
    enu = cfac * hcal_edepthCorrect2;
    enh = cfac * hcal_edepthHot2;
    enc = cfac * hcal_edepthHotCorrect2;
    chgS = hcal_cdepthHot2;
    actL = hcal_activeHotL2;
    chgB = hcal_cdepthHotBG2;
  } else if (dep == 2) {
    enb = cfac * hcal_edepth3;
    enu = cfac * hcal_edepthCorrect3;
    enh = cfac * hcal_edepthHot3;
    enc = cfac * hcal_edepthHotCorrect3;
    chgS = hcal_cdepthHot3;
    actL = hcal_activeHotL3;
    chgB = hcal_cdepthHotBG3;
  } else if (dep == 3) {
    enb = cfac * hcal_edepth4;
    enu = cfac * hcal_edepthCorrect4;
    enh = cfac * hcal_edepthHot4;
    enc = cfac * hcal_edepthHotCorrect4;
    chgS = hcal_cdepthHot4;
    actL = hcal_activeHotL4;
    chgB = hcal_cdepthHotBG4;
  } else if (dep == 4) {
    enb = cfac * hcal_edepth5;
    enu = cfac * hcal_edepthCorrect5;
    enh = cfac * hcal_edepthHot5;
    enc = cfac * hcal_edepthHotCorrect5;
    chgS = hcal_cdepthHot5;
    actL = hcal_activeHotL5;
    chgB = hcal_cdepthHotBG5;
  } else if (dep == 5) {
    if (dep <= maxDepth_) {
      enb = cfac * hcal_edepth6;
      enu = cfac * hcal_edepthCorrect6;
      enh = cfac * hcal_edepthHot6;
      enc = cfac * hcal_edepthHotCorrect6;
      chgS = hcal_cdepthHot6;
      actL = hcal_activeHotL6;
      chgB = hcal_cdepthHotBG6;
    } else {
      enb = enu = enh = enc = chgS = actL = chgB = 0;
      flag = false;
    }
  } else if (dep == 6) {
    if (dep <= maxDepth_) {
      enb = cfac * hcal_edepth7;
      enu = cfac * hcal_edepthCorrect7;
      enh = cfac * hcal_edepthHot7;
      enc = cfac * hcal_edepthHotCorrect7;
      chgS = hcal_cdepthHot7;
      actL = hcal_activeHotL7;
      chgB = hcal_cdepthHotBG7;
    } else {
      enb = enu = enh = enc = chgS = actL = chgB = 0;
      flag = false;
    }
  } else {
    enb = enu = enh = enc = chgS = actL = chgB = 0;
    flag = false;
  }
  return flag;
}

bool HBHEMuonOfflineAnalyzer::looseMuon() {
  if (pt_of_muon > 20.) {
    if (mediumMuon2016()) {
      if (IsolationR04 < 0.25) {
        return true;
      }
    }
  }
  return false;
}

bool HBHEMuonOfflineAnalyzer::softMuon() {
  if (pt_of_muon > 20.) {
    if (mediumMuon2016()) {
      if (IsolationR03 < 0.10) {
        return true;
      }
    }
  }
  return false;
}

bool HBHEMuonOfflineAnalyzer::tightMuon() {
  if (pt_of_muon > 20.) {
    if (mediumMuon2016()) {
      if (IsolationR04 < 0.15) {
        return true;
      }
    }
  }
  return false;
}

bool HBHEMuonOfflineAnalyzer::mediumMuon2016() {
  bool medium16 = (((PF_Muon) && (Global_Muon || Tracker_muon)) && (tight_validFraction > 0.49));
  if (!medium16)
    return medium16;

  bool goodGlob = (Global_Muon && GlobTrack_Chi < 3 && muon_chi2LocalPosition < 12 && muon_trkKink < 20);
  medium16 = muon_segComp > (goodGlob ? 0.303 : 0.451);
  return medium16;
}

void HBHEMuonOfflineAnalyzer::etaPhiHcal(unsigned int detId, int &eta, int &phi, int &depth) {
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

void HBHEMuonOfflineAnalyzer::etaPhiEcal(
    unsigned int detId, int &type, int &zside, int &etaX, int &phiY, int &plane, int &strip) {
  type = ((detId >> 25) & 0x7);
  // std::cout<<"type"<<type<<std::endl;
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

void HBHEMuonOfflineAnalyzer::calculateP(double pt, double eta, double &pM) {
  pM = (pt * cos(2 * (1 / atan(exp(eta)))));
}

void HBHEMuonOfflineAnalyzer::close() {
  output_file->cd();
  std::cout << "file yet to be Written" << std::endl;
  outtree_->Write();
  writeHistograms();
  std::cout << "file Written" << std::endl;
  output_file->Close();
  std::cout << "now doing return" << std::endl;
}

void HBHEMuonOfflineAnalyzer::writeHistograms() {
  //output_file->cd();
  std::string type[] = {"tight", "soft", "loose"};
  char name[128];

  std::cout << "WriteHistograms" << std::endl;
  nHist = 0;
  for (int eta = etaMin_; eta <= etaMax_; ++eta) {
    int nDepth = nDepthBins(eta, -1);
    int nPhi = nPhiBins(eta);
    if (debug_)
      std::cout << "Eta:" << eta << " nDepths " << nDepth << " nPhis " << nPhi << std::endl;
    for (int depth = 0; depth < nDepth; ++depth) {
      if (debug_)
        std::cout << "Eta:" << eta << "Depth:" << depth << std::endl;
      for (int PHI = 0; PHI < nPhi; ++PHI) {
        indxEta[eta - 1][depth][PHI] = nHist;
        nHist += 2;
      }
    }
  }

  TDirectory *d_output_file[29];
  h_evtype->Write();
  //output_file->cd();
  int i(nCut_);
  h_Pt_Muon->Write();
  h_Eta_Muon->Write();
  h_Phi_Muon->Write();

  h_P_Muon->Write();
  h_PF_Muon->Write();
  h_GlobTrack_Chi->Write();
  h_Global_Muon_Hits->Write();

  h_MatchedStations->Write();
  h_Tight_TransImpactparameter->Write();
  h_Tight_LongitudinalImpactparameter->Write();

  h_InnerTrackPixelHits->Write();
  h_TrackerLayer->Write();
  h_IsolationR04->Write();

  h_Global_Muon->Write();
  h_TransImpactParameter->Write();

  h_TransImpactParameterBin1->Write();
  h_TransImpactParameterBin2->Write();

  h_LongImpactParameter->Write();
  h_LongImpactParameterBin1->Write();
  h_LongImpactParameterBin2->Write();

  h_ecal_energy->Write();
  h_hcal_energy->Write();

  h_3x3_ecal->Write();
  h_1x1_hcal->Write();

  h_EtaX_hcal->Write();
  h_PhiY_hcal->Write();

  h_EtaX_ecal->Write();
  h_PhiY_ecal->Write();

  h_Eta_ecal->Write();
  h_Phi_ecal->Write();
  h_MuonHittingEcal->Write();
  h_HotCell->Write();

  output_file->cd();
  for (int eta = etaMin_; eta <= etaMax_; ++eta) {
    int nDepth = nDepthBins(eta, -1);
    int nPhi = nPhiBins(eta);
    sprintf(name, "Dir_muon_type_%s_ieta%d", type[i].c_str(), eta);
    d_output_file[eta - 1] = output_file->mkdir(name);
    //output_file->cd(name);
    d_output_file[eta - 1]->cd();
    for (int depth = 0; depth < nDepth; ++depth) {
      for (int PHI = 0; PHI < nPhi; ++PHI) {
        //	std::cout<<"eta:"<<eta<<"depth:"<<depth<<"PHI:"<<PHI<<std::endl;
        int ih = indxEta[eta - 1][depth][PHI];
        //	std::cout<<"ih:"<<ih<<std::endl;
        h_Hot_MuonEnergy_hcal_HotCell[ih]->Write();

        h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ih]->Write();
        h_active_length_Fill[ih]->Write();
        h_p_muon_ineta[ih]->Write();
        h_charge_signal[ih]->Write();
        h_charge_bg[ih]->Write();
        ih++;
        h_Hot_MuonEnergy_hcal_HotCell[ih]->Write();
        h_Hot_MuonEnergy_hcal_HotCell_VsActiveLength[ih]->Write();
        h_active_length_Fill[ih]->Write();
        h_p_muon_ineta[ih]->Write();
        h_charge_signal[ih]->Write();
        h_charge_bg[ih]->Write();
      }
    }
    output_file->cd();
  }
  output_file->cd();
}

int HBHEMuonOfflineAnalyzer::nDepthBins(int eta, int phi) {
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

int HBHEMuonOfflineAnalyzer::nPhiBins(int eta) {
  int nphi = (eta <= 20) ? 72 : 36;
  if (modeLHC_ == 5 && eta > 16)
    nphi = 360;
  return nphi;
}

float HBHEMuonOfflineAnalyzer::getCorr(int run, unsigned int id) {
  float cfac(1.0);
  int ip(-1);
  for (unsigned int k = 0; k < runlow_.size(); ++k) {
    unsigned int i = runlow_.size() - k - 1;
    if (run >= runlow_[i]) {
      ip = (int)(i);
      break;
    }
  }
  if (debug_) {
    std::cout << "Run " << run << " Perdiod " << ip << std::endl;
  }
  unsigned idx = correctDetId(id);
  if (ip >= 0) {
    std::map<unsigned int, float>::iterator itr = corrFac_[ip].find(idx);
    if (itr != corrFac_[ip].end())
      cfac = itr->second;
  }
  if (debug_) {
    int subdet, zside, ieta, iphi, depth;
    unpackDetId(idx, subdet, zside, ieta, iphi, depth);
    std::cout << "ID " << std::hex << id << std::dec << " (Sub " << subdet << " eta " << zside * ieta << " phi " << iphi
              << " depth " << depth << ")  Factor " << cfac << std::endl;
  }
  return cfac;
}

std::vector<std::string> HBHEMuonOfflineAnalyzer::splitString(const std::string &fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

unsigned int HBHEMuonOfflineAnalyzer::getDetIdHBHE(int ieta, int iphi, int depth) {
  int eta = std::abs(ieta);
  int subdet(0);
  if (eta > 16)
    subdet = 2;
  else if (eta == 16 && depth > 2)
    subdet = 2;
  else
    subdet = 1;
  return getDetId(subdet, ieta, iphi, depth);
}

unsigned int HBHEMuonOfflineAnalyzer::getDetId(int subdet, int ieta, int iphi, int depth) {
  // All numbers used here are described as masks/offsets in
  // DataFormats/HcalDetId/interface/HcalDetId.h
  unsigned int id = ((4 << 28) | ((subdet & 0x7) << 25));
  id |= ((0x1000000) | ((depth & 0xF) << 20) | ((ieta > 0) ? (0x80000 | (ieta << 10)) : ((-ieta) << 10)) |
         (iphi & 0x3FF));
  return id;
}

unsigned int HBHEMuonOfflineAnalyzer::correctDetId(const unsigned int &detId) {
  int subdet, ieta, zside, depth, iphi;
  unpackDetId(detId, subdet, zside, ieta, iphi, depth);
  if (subdet == 0) {
    if (ieta > 16)
      subdet = 2;
    else if (ieta == 16 && depth > 2)
      subdet = 2;
    else
      subdet = 1;
  }
  unsigned int id = getDetId(subdet, ieta * zside, iphi, depth);
  if ((id != detId) && debug_) {
    std::cout << "Correct Id " << std::hex << detId << " to " << id << std::dec << "(Sub " << subdet << " eta "
              << ieta * zside << " phi " << iphi << " depth " << depth << ")" << std::endl;
  }
  return id;
}

void HBHEMuonOfflineAnalyzer::unpackDetId(
    unsigned int detId, int &subdet, int &zside, int &ieta, int &iphi, int &depth) {
  // The maskings are defined in DataFormats/DetId/interface/DetId.h
  //                      and in DataFormats/HcalDetId/interface/HcalDetId.h
  // The macro does not invoke the classes there and use them
  subdet = ((detId >> 25) & (0x7));
  if ((detId & 0x1000000) == 0) {
    ieta = ((detId >> 7) & 0x3F);
    zside = (detId & 0x2000) ? (1) : (-1);
    depth = ((detId >> 14) & 0x1F);
    iphi = (detId & 0x3F);
  } else {
    ieta = ((detId >> 10) & 0x1FF);
    zside = (detId & 0x80000) ? (1) : (-1);
    depth = ((detId >> 20) & 0xF);
    iphi = (detId & 0x3FF);
  }
}
