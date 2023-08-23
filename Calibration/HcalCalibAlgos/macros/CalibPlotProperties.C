//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibPlotProperties.C+g
//  CalibPlotProperties c1(fname, dirname, dupFileName, prefix, corrFileName,
//	                   rcorFileName, puCorr, flag, dataMC, truncateFlag,
//                         useGen, scale, useScale, etalo, etahi, runlo, runhi,
//                         phimin, phimax, zside, nvxlo, nvxhi, rbxFile,
//                         exclude, etamax);
//  c1.Loop(nentries);
//  c1.savePlot(histFileName, append, all, debug);
//
//        This will prepare a set of histograms with properties of the tracks
//        which can be displayed by the method in this file
//
//  PlotHist(histFileName, prefix, text, flagC, etalo, etahi, save)
//
//        This will plot the heistograms and save the canvases
//
// .L CalibPlotProperties.C+g
//  CalibSplit c1(fname, dirname, outFileName, pmin, pmax, debug);
//  c1.Loop(nentries);
//
//   where:
//
//   fname   (const char*)     = file name of the input ROOT tree
//                               or name of the file containing a list of
//                               file names of input ROOT trees
//   dirname (const char*)     = name of the directory where Tree resides
//                               (use "HcalIsoTrkAnalyzer")
//   dupFileName (const char*) = name of the file containing list of entries
//                               of duplicate events
//   prefix (std::string)      = String to be added to the name of histogram
//                               (usually a 4 character string; default="")
//   corrFileName (const char*)= name of the text file having the correction
//                               factors to be used (default="", no corr.)
//   rcorFileName (const char*)= name of the text file having the correction
//                               factors as a function of run numbers or depth
//                               to be used for raddam/depth/pileup/phisym
//                               dependent correction  (default="", no corr.)
//   puCorr (int)              = PU correction to be applied or not: 0 no
//                               correction; < 0 use eDelta; > 0 rho dependent
//                               correction (-8)
//   flag (int)                = 7 digit integer (ymlthdo) with control
//                               information (y=2/1/0 containing list of
//                               ieta, iphi of channels to be selected (2);
//                               list containing depth dependent weights for
//                               each ieta (1); list of duplicate entries
//                               (0) in dupFileName; m=0/1 for controlling
//                               creation of depth depedendent histograms;
//                               l=4/3/2/1/0 for type of rcorFileName (4 for
//                               using results from phi-symmetry; 3 for
//                               pileup correction using machine learning
//                               method; 2 for overall response corrections;
//                               1 for depth dependence corrections; 0 for
//                               raddam corrections);
//                               t = bit information (lower bit set will
//                               apply a cut on L1 closeness; and higher bit
//                               set read correction file with Marina format);
//                               h =0/1 flag to create energy histograms
//                               d =0/1 flag to create basic set of histograms;
//                               o =0/1/2 for tight / loose / flexible
//                               selection). Default = 101111
//   dataMC (bool)             = true/false for data/MC (default true)
//   truncateFlag    (int)     = Flag to treat different depths differently (0)
//                               both depths of ieta 15, 16 of HB as depth 1 (1)
//                               all depths as depth 1 (2), all depths in HE
//                               with values > 1 as depth 2 (3), all depths in
//                               HB with values > 1 as depth 2 (4), all depths
//                               in HB and HE with values > 1 as depth 2 (5)
//                               (Default 0)
//   useGen (bool)             = true/false to use generator level momentum
//                               or reconstruction level momentum (def false)
//   scale (double)            = energy scale if correction factor to be used
//                               (default = 1.0)
//   useScale (int)            = two digit number (do) with o: as the flag for
//                               application of scale factor (0: nowehere,
//                               1: barrel; 2: endcap, 3: everywhere)
//                               barrel => |ieta| < 16; endcap => |ieta| > 15;
//                               d: as the format for threshold application,
//                               0: no threshold; 1: 2022 prompt data; 2:
//                               2022 reco data; 3: 2023 prompt data
//                               (default = 0)
//   etalo/etahi (int,int)     = |eta| ranges (0:30)
//   runlo  (int)              = lower value of run number to be included (+ve)
//                               or excluded (-ve) (default 0)
//   runhi  (int)              = higher value of run number to be included
//                               (+ve) or excluded (-ve) (def 9999999)
//   phimin          (int)     = minimum iphi value (1)
//   phimax          (int)     = maximum iphi value (72)
//   zside           (int)     = the side of the detector if phimin and phimax
//                               differ from 1-72 (1)
//   nvxlo           (int)     = minimum # of vertex in event to be used (0)
//   nvxhi           (int)     = maximum # of vertex in event to be used (1000)
//   rbxFile         (char *)  = Name of the file containing a list of RBX's
//                               to be consdered (default = ""). RBX's are
//                               specified by zside*(Subdet*100+RBX #).
//                               For HEP17 it will be 217
//   exclude         (bool)    = RBX specified by the contents in *rbxFile* to
//                               be exluded or only considered (default = false)
//   etamax          (bool)    = if set and if the corr-factor not found in the
//                               corrFactor table, the corr-factor for the
//                               corresponding zside, depth=1 and maximum ieta
//                               in the table is taken (false)
//   nentries        (int)     = maximum number of entries to be processed,
//                               if -1, all entries to be processed (-1)
//
//   histFileName (std::string)= name of the file containing saved histograms
//   append (bool)             = true/false if the histogram file to be opened
//                               in append/output mode
//   all (bool)                = true/false if all histograms to be saved or
//                               not (def false)
//
//   text  (string)            = string to be put in the title
//   flagC (int)               = 3 digit integer (hdo) with control
//                               information (h=0/1 for plottting the depth
//                               depedendent histograms;
//                               d =0/1 flag to plot energy histograms;
//                               o =0/1 flag to plot basic set of histograms;
//                               Default = 111
//   save (int)                = flag to create or not save the plot in a file
//                               (0 = no save, > 0 pdf file, < 0 hpg file)
//
//   outFileName (std::string)= name of the file containing saved tree
//   pmin (double)            = minimum track momentum (40.0)
//   pmax (double)            = maximum track momentum (60.0)
//   debug (bool)             = debug flag (false)
//////////////////////////////////////////////////////////////////////////////
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

void unpackDetId(unsigned int, int &, int &, int &, int &, int &);
#include "CalibCorr.C"

namespace CalibPlots {
  static const int npbin = 4;
  static const int netabin = 4;
  static const int ndepth = 7;
  static const int ntitles = 5;
  static const int npbin0 = (npbin + 1);
  int getP(int k) {
    const int ipbin[npbin0] = {20, 30, 40, 60, 100};
    return ((k >= 0 && k < npbin0) ? ipbin[k] : 0);
  }
  double getMomentum(int k) { return (double)(getP(k)); }
  int getEta(int k) {
    int ietas[netabin] = {0, 13, 18, 23};
    return ((k >= 0 && k < netabin) ? ietas[k] : -1);
  }
  std::string getTitle(int k) {
    std::string titl[ntitles] = {
        "all tracks", "good quality tracks", "selected tracks", "isolated good tracks", "tracks having MIP in ECAL"};
    return ((k >= 0 && k < ntitles) ? titl[k] : "");
  }
}  // namespace CalibPlots

class CalibPlotProperties {
public:
  TChain *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t t_Run;
  Int_t t_Event;
  Int_t t_DataType;
  Int_t t_ieta;
  Int_t t_iphi;
  Double_t t_EventWeight;
  Int_t t_nVtx;
  Int_t t_nTrk;
  Int_t t_goodPV;
  Double_t t_l1pt;
  Double_t t_l1eta;
  Double_t t_l1phi;
  Double_t t_l3pt;
  Double_t t_l3eta;
  Double_t t_l3phi;
  Double_t t_p;
  Double_t t_pt;
  Double_t t_phi;
  Double_t t_mindR1;
  Double_t t_mindR2;
  Double_t t_eMipDR;
  Double_t t_eHcal;
  Double_t t_eHcal10;
  Double_t t_eHcal30;
  Double_t t_hmaxNearP;
  Double_t t_rhoh;
  Bool_t t_selectTk;
  Bool_t t_qltyFlag;
  Bool_t t_qltyMissFlag;
  Bool_t t_qltyPVFlag;
  Double_t t_gentrackP;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double> *t_HitEnergies;
  std::vector<bool> *t_trgbits;
  std::vector<unsigned int> *t_DetIds1;
  std::vector<unsigned int> *t_DetIds3;
  std::vector<double> *t_HitEnergies1;
  std::vector<double> *t_HitEnergies3;

  // List of branches
  TBranch *b_t_Run;           //!
  TBranch *b_t_Event;         //!
  TBranch *b_t_DataType;      //!
  TBranch *b_t_ieta;          //!
  TBranch *b_t_iphi;          //!
  TBranch *b_t_EventWeight;   //!
  TBranch *b_t_nVtx;          //!
  TBranch *b_t_nTrk;          //!
  TBranch *b_t_goodPV;        //!
  TBranch *b_t_l1pt;          //!
  TBranch *b_t_l1eta;         //!
  TBranch *b_t_l1phi;         //!
  TBranch *b_t_l3pt;          //!
  TBranch *b_t_l3eta;         //!
  TBranch *b_t_l3phi;         //!
  TBranch *b_t_p;             //!
  TBranch *b_t_pt;            //!
  TBranch *b_t_phi;           //!
  TBranch *b_t_mindR1;        //!
  TBranch *b_t_mindR2;        //!
  TBranch *b_t_eMipDR;        //!
  TBranch *b_t_eHcal;         //!
  TBranch *b_t_eHcal10;       //!
  TBranch *b_t_eHcal30;       //!
  TBranch *b_t_hmaxNearP;     //!
  TBranch *b_t_rhoh;          //!
  TBranch *b_t_selectTk;      //!
  TBranch *b_t_qltyFlag;      //!
  TBranch *b_t_qltyMissFlag;  //!
  TBranch *b_t_qltyPVFlag;    //!
  TBranch *b_t_gentrackP;     //!
  TBranch *b_t_DetIds;        //!
  TBranch *b_t_HitEnergies;   //!
  TBranch *b_t_trgbits;       //!
  TBranch *b_t_DetIds1;       //!
  TBranch *b_t_DetIds3;       //!
  TBranch *b_t_HitEnergies1;  //!
  TBranch *b_t_HitEnergies3;  //!

  CalibPlotProperties(const char *fname,
                      const std::string &dirname,
                      const char *dupFileName,
                      const std::string &prefix = "",
                      const char *corrFileName = "",
                      const char *rcorFileName = "",
                      int puCorr = -8,
                      int flag = 101111,
                      bool dataMC = true,
                      int truncateFlag = 0,
                      bool useGen = false,
                      double scale = 1.0,
                      int useScale = 0,
                      int etalo = 0,
                      int etahi = 30,
                      int runlo = 0,
                      int runhi = 99999999,
                      int phimin = 1,
                      int phimax = 72,
                      int zside = 1,
                      int nvxlo = 0,
                      int nvxhi = 1000,
                      const char *rbxFile = "",
                      bool exclude = false,
                      bool etamax = false);
  virtual ~CalibPlotProperties();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *);
  virtual void Loop(Long64_t nentries = -1);
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  bool goodTrack(double &eHcal, double &cut, bool debug);
  bool selectPhi(bool debug);
  void savePlot(const std::string &theName, bool append, bool all = false, bool debug = false);
  void correctEnergy(double &ener);

private:
  static const int kp50 = 2;
  CalibCorrFactor *corrFactor_;
  CalibCorr *cFactor_;
  CalibSelectRBX *cSelect_;
  CalibDuplicate *cDuplicate_;
  const std::string fname_, dirnm_, prefix_, outFileName_;
  const int corrPU_, flag_;
  const bool dataMC_, useGen_;
  const int truncateFlag_;
  const int etalo_, etahi_;
  int runlo_, runhi_;
  const int phimin_, phimax_, zside_, nvxlo_, nvxhi_;
  const char *rbxFile_;
  bool exclude_, corrE_, cutL1T_;
  bool includeRun_, getHist_;
  int flexibleSelect_, ifDepth_, duplicate_, thrForm_;
  bool plotBasic_, plotEnergy_, plotHists_;
  double log2by18_;
  std::ofstream fileout_;
  std::vector<std::pair<int, int> > events_;
  TH1D *h_p[CalibPlots::ntitles];
  TH1D *h_eta[CalibPlots::ntitles], *h_nvtx, *h_nvtxEv, *h_nvtxTk;
  std::vector<TH1D *> h_eta0, h_eta1, h_eta2, h_eta3, h_eta4;
  std::vector<TH1D *> h_dL1, h_vtx;
  std::vector<TH1D *> h_etaEH[CalibPlots::npbin0];
  std::vector<TH1D *> h_etaEp[CalibPlots::npbin0];
  std::vector<TH1D *> h_etaEE[CalibPlots::npbin0];
  std::vector<TH1D *> h_etaEE0[CalibPlots::npbin0];
  std::vector<TH1D *> h_mom, h_eEcal, h_eHcal;
  std::vector<TH1F *> h_bvlist, h_bvlist2, h_evlist, h_evlist2;
  std::vector<TH1F *> h_bvlist3, h_evlist3;
  TH2F *h_etaE;
};

CalibPlotProperties::CalibPlotProperties(const char *fname,
                                         const std::string &dirnm,
                                         const char *dupFileName,
                                         const std::string &prefix,
                                         const char *corrFileName,
                                         const char *rcorFileName,
                                         int puCorr,
                                         int flag,
                                         bool dataMC,
                                         int truncate,
                                         bool useGen,
                                         double scl,
                                         int useScale,
                                         int etalo,
                                         int etahi,
                                         int runlo,
                                         int runhi,
                                         int phimin,
                                         int phimax,
                                         int zside,
                                         int nvxlo,
                                         int nvxhi,
                                         const char *rbxFile,
                                         bool exc,
                                         bool etam)
    : corrFactor_(nullptr),
      cFactor_(nullptr),
      cSelect_(nullptr),
      cDuplicate_(nullptr),
      fname_(fname),
      dirnm_(dirnm),
      prefix_(prefix),
      corrPU_(puCorr),
      flag_(flag),
      dataMC_(dataMC),
      useGen_(useGen),
      truncateFlag_(truncate),
      etalo_(etalo),
      etahi_(etahi),
      runlo_(runlo),
      runhi_(runhi),
      phimin_(phimin),
      phimax_(phimax),
      zside_(zside),
      nvxlo_(nvxlo),
      nvxhi_(nvxhi),
      rbxFile_(rbxFile),
      exclude_(exc),
      includeRun_(true) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree

  flexibleSelect_ = ((flag_ / 1) % 10);
  plotBasic_ = (((flag_ / 10) % 10) > 0);
  plotEnergy_ = (((flag_ / 10) % 10) > 0);
  int oneplace = ((flag_ / 1000) % 10);
  cutL1T_ = (oneplace % 2);
  bool marina = ((oneplace / 2) % 2);
  ifDepth_ = ((flag_ / 10000) % 10);
  plotHists_ = (((flag_ / 100000) % 10) > 0);
  duplicate_ = ((flag_ / 1000000) % 10);
  log2by18_ = std::log(2.5) / 18.0;
  if (runlo_ < 0 || runhi_ < 0) {
    runlo_ = std::abs(runlo_);
    runhi_ = std::abs(runhi_);
    includeRun_ = false;
  }
  int useScale0 = useScale % 10;
  thrForm_ = useScale / 10;
  char treeName[400];
  sprintf(treeName, "%s/CalibTree", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << " flags " << flexibleSelect_ << "|"
            << plotBasic_ << "|"
            << "|" << plotEnergy_ << "|" << plotHists_ << "|" << corrPU_ << " cons " << log2by18_ << " eta range "
            << etalo_ << ":" << etahi_ << " run range " << runlo_ << ":" << runhi_ << " (inclusion flag " << includeRun_
            << ") Vertex Range " << nvxlo_ << ":" << nvxhi_ << " Threshold Flag " << thrForm_ << std::endl;
  corrFactor_ = new CalibCorrFactor(corrFileName, useScale0, scl, etam, marina, false);
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain);
    if (std::string(rcorFileName) != "") {
      cFactor_ = new CalibCorr(rcorFileName, ifDepth_, false);
      if (cFactor_->absent())
        ifDepth_ = -1;
    } else {
      ifDepth_ = -1;
    }
    if (std::string(dupFileName) != "")
      cDuplicate_ = new CalibDuplicate(dupFileName, duplicate_, false);
    if (std::string(rbxFile) != "")
      cSelect_ = new CalibSelectRBX(rbxFile, false);
  }
}

CalibPlotProperties::~CalibPlotProperties() {
  delete corrFactor_;
  delete cFactor_;
  delete cSelect_;
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t CalibPlotProperties::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibPlotProperties::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (!fChain->InheritsFrom(TChain::Class()))
    return centry;
  TChain *chain = (TChain *)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void CalibPlotProperties::Init(TChain *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_DetIds = 0;
  t_DetIds1 = 0;
  t_DetIds3 = 0;
  t_HitEnergies = 0;
  t_HitEnergies1 = 0;
  t_HitEnergies3 = 0;
  t_trgbits = 0;
  // Set branch addresses and branch pointers
  fChain = tree;
  fCurrent = -1;
  if (!tree)
    return;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_DataType", &t_DataType, &b_t_DataType);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_iphi", &t_iphi, &b_t_iphi);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_nVtx", &t_nVtx, &b_t_nVtx);
  fChain->SetBranchAddress("t_nTrk", &t_nTrk, &b_t_nTrk);
  fChain->SetBranchAddress("t_goodPV", &t_goodPV, &b_t_goodPV);
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_pt", &t_pt, &b_t_pt);
  fChain->SetBranchAddress("t_phi", &t_phi, &b_t_phi);
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_eHcal10", &t_eHcal10, &b_t_eHcal10);
  fChain->SetBranchAddress("t_eHcal30", &t_eHcal30, &b_t_eHcal30);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_rhoh", &t_rhoh, &b_t_rhoh);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyFlag", &t_qltyFlag, &b_t_qltyFlag);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_gentrackP", &t_gentrackP, &b_t_gentrackP);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  fChain->SetBranchAddress("t_trgbits", &t_trgbits, &b_t_trgbits);
  fChain->SetBranchAddress("t_DetIds1", &t_DetIds1, &b_t_DetIds1);
  fChain->SetBranchAddress("t_DetIds3", &t_DetIds3, &b_t_DetIds3);
  fChain->SetBranchAddress("t_HitEnergies1", &t_HitEnergies1, &b_t_HitEnergies1);
  fChain->SetBranchAddress("t_HitEnergies3", &t_HitEnergies3, &b_t_HitEnergies3);
  Notify();

  char name[20], title[200];
  unsigned int kk(0);

  if (plotBasic_) {
    std::cout << "Book Basic Histos" << std::endl;
    h_nvtx = new TH1D("hnvtx", "Number of vertices (selected entries)", 10, 0, 100);
    h_nvtx->Sumw2();
    h_nvtxEv = new TH1D("hnvtxEv", "Number of vertices (selected events)", 10, 0, 100);
    h_nvtxEv->Sumw2();
    h_nvtxTk = new TH1D("hnvtxTk", "Number of vertices (selected tracks)", 10, 0, 100);
    h_nvtxTk->Sumw2();
    for (int k = 0; k < CalibPlots::ntitles; ++k) {
      sprintf(name, "%sp%d", prefix_.c_str(), k);
      sprintf(title, "Momentum for %s", CalibPlots::getTitle(k).c_str());
      h_p[k] = new TH1D(name, title, 150, 0.0, 150.0);
      sprintf(name, "%seta%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s", CalibPlots::getTitle(k).c_str());
      h_eta[k] = new TH1D(name, title, 60, -30.0, 30.0);
    }
    for (int k = 0; k < CalibPlots::npbin; ++k) {
      sprintf(name, "%seta0%d", prefix_.c_str(), k);
      sprintf(title,
              "#eta for %s (p = %d:%d GeV)",
              CalibPlots::getTitle(0).c_str(),
              CalibPlots::getP(k),
              CalibPlots::getP(k + 1));
      h_eta0.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta0.size() - 1;
      h_eta0[kk]->Sumw2();
      sprintf(name, "%seta1%d", prefix_.c_str(), k);
      sprintf(title,
              "#eta for %s (p = %d:%d GeV)",
              CalibPlots::getTitle(1).c_str(),
              CalibPlots::getP(k),
              CalibPlots::getP(k + 1));
      h_eta1.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta1.size() - 1;
      h_eta1[kk]->Sumw2();
      sprintf(name, "%seta2%d", prefix_.c_str(), k);
      sprintf(title,
              "#eta for %s (p = %d:%d GeV)",
              CalibPlots::getTitle(2).c_str(),
              CalibPlots::getP(k),
              CalibPlots::getP(k + 1));
      h_eta2.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta2.size() - 1;
      h_eta2[kk]->Sumw2();
      sprintf(name, "%seta3%d", prefix_.c_str(), k);
      sprintf(title,
              "#eta for %s (p = %d:%d GeV)",
              CalibPlots::getTitle(3).c_str(),
              CalibPlots::getP(k),
              CalibPlots::getP(k + 1));
      h_eta3.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta3.size() - 1;
      h_eta3[kk]->Sumw2();
      sprintf(name, "%seta4%d", prefix_.c_str(), k);
      sprintf(title,
              "#eta for %s (p = %d:%d GeV)",
              CalibPlots::getTitle(4).c_str(),
              CalibPlots::getP(k),
              CalibPlots::getP(k + 1));
      h_eta4.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta4.size() - 1;
      h_eta4[kk]->Sumw2();
      sprintf(name, "%sdl1%d", prefix_.c_str(), k);
      sprintf(title, "Distance from L1 (p = %d:%d GeV)", CalibPlots::getP(k), CalibPlots::getP(k + 1));
      h_dL1.push_back(new TH1D(name, title, 160, 0.0, 8.0));
      kk = h_dL1.size() - 1;
      h_dL1[kk]->Sumw2();
      sprintf(name, "%svtx%d", prefix_.c_str(), k);
      sprintf(title, "N_{Vertex} (p = %d:%d GeV)", CalibPlots::getP(k), CalibPlots::getP(k + 1));
      h_vtx.push_back(new TH1D(name, title, 100, 0.0, 100.0));
      kk = h_vtx.size() - 1;
      h_vtx[kk]->Sumw2();
    }
  }

  if (plotEnergy_) {
    std::cout << "Make plots for good tracks" << std::endl;
    for (int k = 0; k < CalibPlots::npbin; ++k) {
      for (int j = etalo_; j <= etahi_ + 1; ++j) {
        sprintf(name, "%senergyH%d%d", prefix_.c_str(), k, j);
        if (j > etahi_)
          sprintf(title,
                  "HCAL energy for %s (p = %d:%d GeV |#eta| = %d:%d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getP(k),
                  CalibPlots::getP(k + 1),
                  etalo_,
                  etahi_);
        else
          sprintf(title,
                  "HCAL energy for %s (p = %d:%d GeV |#eta| = %d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getP(k),
                  CalibPlots::getP(k + 1),
                  j);
        h_etaEH[k].push_back(new TH1D(name, title, 200, 0, 200));
        kk = h_etaEH[k].size() - 1;
        h_etaEH[k][kk]->Sumw2();
        sprintf(name, "%senergyP%d%d", prefix_.c_str(), k, j);
        if (j > etahi_)
          sprintf(title,
                  "momentum for %s (p = %d:%d GeV |#eta| = %d:%d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getP(k),
                  CalibPlots::getP(k + 1),
                  etalo_,
                  etahi_);
        else
          sprintf(title,
                  "momentum for %s (p = %d:%d GeV |#eta| = %d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getP(k),
                  CalibPlots::getP(k + 1),
                  j);
        h_etaEp[k].push_back(new TH1D(name, title, 100, 0, 100));
        kk = h_etaEp[k].size() - 1;
        h_etaEp[k][kk]->Sumw2();
        sprintf(name, "%senergyE%d%d", prefix_.c_str(), k, j);
        if (j > etahi_)
          sprintf(title,
                  "ECAL energy for %s (p = %d:%d GeV |#eta| = %d:%d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getP(k),
                  CalibPlots::getP(k + 1),
                  etalo_,
                  etahi_);
        else
          sprintf(title,
                  "ECAL energy for %s (p = %d:%d GeV |#eta| = %d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getP(k),
                  CalibPlots::getP(k + 1),
                  j);
        h_etaEE[k].push_back(new TH1D(name, title, 100, 0, 10));
        kk = h_etaEE[k].size() - 1;
        h_etaEE[k][kk]->Sumw2();
        sprintf(name, "%senergyER%d%d", prefix_.c_str(), k, j);
        h_etaEE0[k].push_back(new TH1D(name, title, 100, 0, 1));
        kk = h_etaEE0[k].size() - 1;
        h_etaEE0[k][kk]->Sumw2();
      }
    }

    for (int j = 0; j < CalibPlots::netabin; ++j) {
      sprintf(name, "%senergyH%d", prefix_.c_str(), j);
      if (j == 0)
        sprintf(title, "HCAL energy for %s (All)", CalibPlots::getTitle(3).c_str());
      else
        sprintf(title,
                "HCAL energy for %s (|#eta| = %d:%d)",
                CalibPlots::getTitle(3).c_str(),
                CalibPlots::getEta(j - 1),
                CalibPlots::getEta(j));
      h_eHcal.push_back(new TH1D(name, title, 200, 0, 200));
      kk = h_eHcal.size() - 1;
      h_eHcal[kk]->Sumw2();
      sprintf(name, "%senergyP%d", prefix_.c_str(), j);
      if (j == 0)
        sprintf(title, "Track momentum for %s (All)", CalibPlots::getTitle(3).c_str());
      else
        sprintf(title,
                "Track momentum for %s (|#eta| = %d:%d)",
                CalibPlots::getTitle(3).c_str(),
                CalibPlots::getEta(j - 1),
                CalibPlots::getEta(j));
      h_mom.push_back(new TH1D(name, title, 100, 0, 100));
      kk = h_mom.size() - 1;
      h_mom[kk]->Sumw2();
      sprintf(name, "%senergyE%d", prefix_.c_str(), j);
      if (j == 0)
        sprintf(title, "ECAL energy for %s (All)", CalibPlots::getTitle(3).c_str());
      else
        sprintf(title,
                "ECAL energy for %s (|#eta| = %d:%d)",
                CalibPlots::getTitle(3).c_str(),
                CalibPlots::getEta(j - 1),
                CalibPlots::getEta(j));
      h_eEcal.push_back(new TH1D(name, title, 100, 0, 10));
      kk = h_eEcal.size() - 1;
      h_eEcal[kk]->Sumw2();
    }
  }

  if (plotHists_) {
    for (int i = 0; i < CalibPlots::ndepth; i++) {
      sprintf(name, "b_edepth%d", i);
      sprintf(title, "Total RecHit energy in depth %d (Barrel)", i + 1);
      h_bvlist.push_back(new TH1F(name, title, 1000, 0, 100));
      h_bvlist[i]->Sumw2();
      sprintf(name, "b_recedepth%d", i);
      sprintf(title, "RecHit energy in depth %d (Barrel)", i + 1);
      h_bvlist2.push_back(new TH1F(name, title, 1000, 0, 100));
      h_bvlist2[i]->Sumw2();
      sprintf(name, "b_nrecdepth%d", i);
      sprintf(title, "#RecHits in depth %d (Barrel)", i + 1);
      h_bvlist3.push_back(new TH1F(name, title, 1000, 0, 100));
      h_bvlist3[i]->Sumw2();
      sprintf(name, "e_edepth%d", i);
      sprintf(title, "Total RecHit energy in depth %d (Endcap)", i + 1);
      h_evlist.push_back(new TH1F(name, title, 1000, 0, 100));
      h_evlist[i]->Sumw2();
      sprintf(name, "e_recedepth%d", i);
      sprintf(title, "RecHit energy in depth %d (Endcap)", i + 1);
      h_evlist2.push_back(new TH1F(name, title, 1000, 0, 100));
      h_evlist2[i]->Sumw2();
      sprintf(name, "e_nrecdepth%d", i);
      sprintf(title, "#RecHits in depth %d (Endcap)", i + 1);
      h_evlist3.push_back(new TH1F(name, title, 1000, 0, 100));
      h_evlist3[i]->Sumw2();
    }
    h_etaE = new TH2F("heta", "", 60, -30, 30, 100, 0, 100);
  }
}

Bool_t CalibPlotProperties::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibPlotProperties::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t CalibPlotProperties::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibPlotProperties::Loop(Long64_t nentries) {
  //   In a ROOT session, you can do:
  //      Root > .L CalibMonitor.C
  //      Root > CalibMonitor t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //   This is the loop skeleton where:
  //      jentry is the global entry number in the chain
  //      ientry is the entry number in the current Tree
  //   Note that the argument to GetEntry must be:
  //      jentry for TChain::GetEntry
  //      ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  const bool debug(false);

  if (fChain == 0)
    return;

  // Find list of duplicate events
  if (nentries < 0)
    nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << ":" << fChain->GetEntriesFast() << std::endl;
  std::vector<std::pair<int, int> > runEvent;
  Long64_t nbytes(0), nb(0);
  unsigned int duplicate(0), good(0), kount(0);
  double sel(0), selHB(0), selHE(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (jentry % 1000000 == 0)
      std::cout << "Entry " << jentry << " Run " << t_Run << " Event " << t_Event << std::endl;
    bool select = ((cDuplicate_ != nullptr) && (duplicate_ == 0)) ? (cDuplicate_->isDuplicate(jentry)) : true;
    if (!select) {
      ++duplicate;
      if (debug)
        std::cout << "Duplicate event " << t_Run << " " << t_Event << " " << t_p << std::endl;
      continue;
    }
    bool selRun = (includeRun_ ? ((t_Run >= runlo_) && (t_Run <= runhi_)) : ((t_Run < runlo_) || (t_Run > runhi_)));
    select =
        (selRun && (fabs(t_ieta) >= etalo_) && (fabs(t_ieta) <= etahi_) && (t_nVtx >= nvxlo_) && (t_nVtx <= nvxhi_));
    if (!select) {
      if (debug)
        std::cout << "Run # " << t_Run << " out of range of " << runlo_ << ":" << runhi_ << " or ieta " << t_ieta
                  << " (" << etalo_ << ":" << etahi_ << ") or nvtx " << t_nVtx << " (" << nvxlo_ << ":" << nvxhi_
                  << ") out of range" << std::endl;
      continue;
    }
    if (cSelect_ != nullptr) {
      if (exclude_) {
        if (cSelect_->isItRBX(t_DetIds))
          continue;
      } else {
        if (!(cSelect_->isItRBX(t_ieta, t_iphi)))
          continue;
      }
    }
    if (cDuplicate_ != nullptr) {
      if (cDuplicate_->select(t_ieta, t_iphi))
        continue;
    }
    select = (!cutL1T_ || (t_mindR1 >= 0.5));
    if (!select) {
      if (debug)
        std::cout << "Reject Run # " << t_Run << " Event # " << t_Event << " too close to L1 trigger " << t_mindR1
                  << std::endl;
      continue;
    }
    select = ((events_.size() == 0) ||
              (std::find(events_.begin(), events_.end(), std::pair<int, int>(t_Run, t_Event)) != events_.end()));
    if (!select) {
      if (debug)
        std::cout << "Reject Run # " << t_Run << " Event # " << t_Event << " not in the selection list" << std::endl;
      continue;
    }

    if (plotBasic_) {
      h_nvtx->Fill(t_nVtx);
      std::pair<int, int> runEv(t_Run, t_Event);
      if (std::find(runEvent.begin(), runEvent.end(), runEv) == runEvent.end()) {
        h_nvtxEv->Fill(t_nVtx);
        runEvent.push_back(runEv);
      }
    }

    // if (Cut(ientry) < 0) continue;
    int kp = CalibPlots::npbin0;
    double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
    for (int k = 1; k < CalibPlots::npbin0; ++k) {
      if (pmom >= CalibPlots::getMomentum(k - 1) && pmom < CalibPlots::getMomentum(k)) {
        kp = k - 1;
        break;
      }
    }
    int jp1 = (((std::abs(t_ieta) >= etalo_) && (std::abs(t_ieta) <= etahi_)) ? (std::abs(t_ieta) - etalo_) : -1);
    int jp2 = (etahi_ - etalo_ + 1);
    int je1 = CalibPlots::netabin;
    for (int j = 1; j < CalibPlots::netabin; ++j) {
      if (std::abs(t_ieta) > CalibPlots::getEta(j - 1) && std::abs(t_ieta) <= CalibPlots::getEta(j)) {
        je1 = j;
        break;
      }
    }
    int je2 = (je1 != CalibPlots::netabin) ? 0 : -1;
    if (debug)
      std::cout << "Bin " << kp << ":" << je1 << ":" << je2 << ":" << jp1 << ":" << jp2 << std::endl;
    double cut = (pmom > 20) ? ((flexibleSelect_ == 0) ? 2.0 : 10.0) : 0.0;
    double rcut(-1000.0);

    // Selection of good track and energy measured in Hcal
    double eHcal(t_eHcal);
    if (corrFactor_->doCorr()) {
      eHcal = 0;
      for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
        // Apply thresholds if necessary
        bool okcell = (thrForm_ == 0) || ((*t_HitEnergies)[k] > threshold((*t_DetIds)[k], thrForm_));
        if (okcell) {
          // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
          unsigned int id = truncateId((*t_DetIds)[k], truncateFlag_, false);
          double cfac = corrFactor_->getCorr(id);
          if ((cFactor_ != 0) && (ifDepth_ != 3) && (ifDepth_ > 0))
            cfac *= cFactor_->getCorr(t_Run, (*t_DetIds)[k]);
          if ((cDuplicate_ != nullptr) && (cDuplicate_->doCorr()))
            cfac *= cDuplicate_->getWeight((*t_DetIds)[k]);
          eHcal += (cfac * ((*t_HitEnergies)[k]));
          if (debug) {
            int subdet, zside, ieta, iphi, depth;
            unpackDetId(id, subdet, zside, ieta, iphi, depth);
            std::cout << zside << ":" << ieta << ":" << depth << " Corr " << cfac << " " << (*t_HitEnergies)[k]
                      << " Out " << eHcal << std::endl;
          }
        }
      }
    }
    bool goodTk = goodTrack(eHcal, cut, debug);
    bool selPhi = selectPhi(debug);
    double rat = (pmom > 0) ? (eHcal / (pmom - t_eMipDR)) : 1.0;
    if (debug)
      std::cout << "Entry " << jentry << " p|eHcal|ratio " << pmom << "|" << t_eHcal << "|" << eHcal << "|" << rat
                << "|" << kp << " Cuts " << t_qltyFlag << "|" << t_selectTk << "|" << (t_hmaxNearP < cut) << "|"
                << (t_eMipDR < 1.0) << "|" << goodTk << "|" << (rat > rcut) << " Select Phi " << selPhi << std::endl;
    if (plotBasic_) {
      h_nvtxTk->Fill(t_nVtx);
      h_p[0]->Fill(pmom, t_EventWeight);
      h_eta[0]->Fill(t_ieta, t_EventWeight);
      if (kp < CalibPlots::npbin0)
        h_eta0[kp]->Fill(t_ieta, t_EventWeight);
      if (t_qltyFlag) {
        h_p[1]->Fill(pmom, t_EventWeight);
        h_eta[1]->Fill(t_ieta, t_EventWeight);
        if (kp < CalibPlots::npbin0)
          h_eta1[kp]->Fill(t_ieta, t_EventWeight);
        if (t_selectTk) {
          h_p[2]->Fill(pmom, t_EventWeight);
          h_eta[2]->Fill(t_ieta, t_EventWeight);
          if (kp < CalibPlots::npbin0)
            h_eta2[kp]->Fill(t_ieta, t_EventWeight);
          if (t_hmaxNearP < cut) {
            h_p[3]->Fill(pmom, t_EventWeight);
            h_eta[3]->Fill(t_ieta, t_EventWeight);
            if (kp < CalibPlots::npbin0)
              h_eta3[kp]->Fill(t_ieta, t_EventWeight);
            if (t_eMipDR < 1.0) {
              h_p[4]->Fill(pmom, t_EventWeight);
              h_eta[4]->Fill(t_ieta, t_EventWeight);
              if (kp < CalibPlots::npbin0) {
                h_eta4[kp]->Fill(t_ieta, t_EventWeight);
                h_dL1[kp]->Fill(t_mindR1, t_EventWeight);
                h_vtx[kp]->Fill(t_goodPV, t_EventWeight);
              }
            }
          }
        }
      }
    }

    if (goodTk && kp != CalibPlots::npbin0 && selPhi) {
      if (rat > rcut) {
        if (plotEnergy_) {
          if (jp1 >= 0) {
            h_etaEH[kp][jp1]->Fill(eHcal, t_EventWeight);
            h_etaEH[kp][jp2]->Fill(eHcal, t_EventWeight);
            h_etaEp[kp][jp1]->Fill(pmom, t_EventWeight);
            h_etaEp[kp][jp2]->Fill(pmom, t_EventWeight);
            h_etaEE[kp][jp1]->Fill(t_eMipDR, t_EventWeight);
            h_etaEE[kp][jp2]->Fill(t_eMipDR, t_EventWeight);
            h_etaEE0[kp][jp1]->Fill(t_eMipDR, t_EventWeight);
            h_etaEE0[kp][jp2]->Fill(t_eMipDR, t_EventWeight);
          }
          if (kp == kp50) {
            if (je1 != CalibPlots::netabin) {
              h_eHcal[je1]->Fill(eHcal, t_EventWeight);
              h_eHcal[je2]->Fill(eHcal, t_EventWeight);
              h_mom[je1]->Fill(pmom, t_EventWeight);
              h_mom[je2]->Fill(pmom, t_EventWeight);
              h_eEcal[je1]->Fill(t_eMipDR, t_EventWeight);
              h_eEcal[je2]->Fill(t_eMipDR, t_EventWeight);
            }
          }
        }

        if (plotHists_) {
          if ((std::fabs(rat - 1) < 0.15) && (kp == kp50) && ((std::abs(t_ieta) < 15) || (std::abs(t_ieta) > 17))) {
            float weight = (dataMC_ ? t_EventWeight : t_EventWeight * puweight(t_nVtx));
            h_etaE->Fill(t_ieta, eHcal, weight);
            sel += weight;
            std::vector<float> bv(7, 0.0f), ev(7, 0.0f);
            std::vector<int> bnrec(7, 0), enrec(7, 0);
            double eb(0), ee(0);
            for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
              // Apply thresholds if necessary
              bool okcell = (thrForm_ == 0) || ((*t_HitEnergies)[k] > threshold((*t_DetIds)[k], thrForm_));
              if (okcell) {
                unsigned int id = truncateId((*t_DetIds)[k], truncateFlag_, false);
                double cfac = corrFactor_->getCorr(id);
                if ((cFactor_ != 0) && (ifDepth_ != 3) && (ifDepth_ > 0))
                  cfac *= cFactor_->getCorr(t_Run, (*t_DetIds)[k]);
                if ((cDuplicate_ != nullptr) && (cDuplicate_->doCorr()))
                  cfac *= cDuplicate_->getWeight((*t_DetIds)[k]);
                double ener = cfac * (*t_HitEnergies)[k];
                if (corrPU_)
                  correctEnergy(ener);
                unsigned int idx = (unsigned int)((*t_DetIds)[k]);
                int subdet, zside, ieta, iphi, depth;
                unpackDetId(idx, subdet, zside, ieta, iphi, depth);
                if (depth > 0 && depth <= CalibPlots::ndepth) {
                  if (subdet == 1) {
                    eb += ener;
                    bv[depth - 1] += ener;
                    h_bvlist2[depth - 1]->Fill(ener, weight);
                    ++bnrec[depth - 1];
                  } else if (subdet == 2) {
                    ee += ener;
                    ev[depth - 1] += ener;
                    h_evlist2[depth - 1]->Fill(ener, weight);
                    ++enrec[depth - 1];
                  }
                }
              }
            }
            bool barrel = (eb > ee);
            if (barrel)
              selHB += weight;
            else
              selHE += weight;
            for (int i = 0; i < CalibPlots::ndepth; i++) {
              if (barrel) {
                h_bvlist[i]->Fill(bv[i], weight);
                h_bvlist3[i]->Fill((bnrec[i] + 0.001), weight);
              } else {
                h_evlist[i]->Fill(ev[i], weight);
                h_evlist3[i]->Fill((enrec[i] + 0.001), weight);
              }
            }
          }
        }
      }
      good++;
    }
    ++kount;
  }
  std::cout << "Finds " << duplicate << " Duplicate events out of " << kount << " events in this file and " << good
            << " selected events" << std::endl;
  if (plotHists_)
    std::cout << "Number of weighted selected events " << sel << " HB " << selHB << " HE " << selHE << std::endl;
}

bool CalibPlotProperties::goodTrack(double &eHcal, double &cuti, bool debug) {
  bool select(true);
  double cut(cuti);
  if (debug) {
    std::cout << "goodTrack input " << eHcal << ":" << cut;
  }
  if (flexibleSelect_ > 1) {
    double eta = (t_ieta > 0) ? t_ieta : -t_ieta;
    cut = 8.0 * exp(eta * log2by18_);
  }
  correctEnergy(eHcal);
  select = ((t_qltyFlag) && (t_selectTk) && (t_hmaxNearP < cut) && (t_eMipDR < 100.0));
  if (debug) {
    std::cout << " output " << eHcal << ":" << cut << ":" << select << std::endl;
  }
  return select;
}

bool CalibPlotProperties::selectPhi(bool debug) {
  bool select(true);
  if (phimin_ > 1 || phimax_ < 72) {
    double eTotal(0), eSelec(0);
    // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
    for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
      // Apply thresholds if necessary
      bool okcell = (thrForm_ == 0) || ((*t_HitEnergies)[k] > threshold((*t_DetIds)[k], thrForm_));
      if (okcell) {
        int iphi = ((*t_DetIds)[k]) & (0x3FF);
        int zside = ((*t_DetIds)[k] & 0x80000) ? (1) : (-1);
        eTotal += ((*t_HitEnergies)[k]);
        if (iphi >= phimin_ && iphi <= phimax_ && zside == zside_)
          eSelec += ((*t_HitEnergies)[k]);
      }
    }
    if (eSelec < 0.9 * eTotal)
      select = false;
    if (debug) {
      std::cout << "Etotal " << eTotal << " and ESelec " << eSelec << " (phi " << phimin_ << ":" << phimax_ << " z "
                << zside_ << ") Selection " << select << std::endl;
    }
  }
  return select;
}

void CalibPlotProperties::savePlot(const std::string &theName, bool append, bool all, bool debug) {
  TFile *theFile(0);
  if (append) {
    theFile = new TFile(theName.c_str(), "UPDATE");
  } else {
    theFile = new TFile(theName.c_str(), "RECREATE");
  }

  theFile->cd();

  if (plotBasic_) {
    if (debug)
      std::cout << "nvtx " << h_nvtx << ":" << h_nvtxEv << ":" << h_nvtxTk << std::endl;
    h_nvtx->Write();
    h_nvtxEv->Write();
    h_nvtxTk->Write();
    for (int k = 0; k < CalibPlots::ntitles; ++k) {
      if (debug)
        std::cout << "[" << k << "] p " << h_p[k] << " eta " << h_eta[k] << std::endl;
      h_p[k]->Write();
      h_eta[k]->Write();
    }
    for (int k = 0; k < CalibPlots::npbin; ++k) {
      if (debug)
        std::cout << "[" << k << "] eta " << h_eta0[k] << ":" << h_eta1[k] << ":" << h_eta2[k] << ":" << h_eta3[k]
                  << ":" << h_eta4[k] << " dl " << h_dL1[k] << " vtx " << h_vtx[k] << std::endl;
      if (h_eta0[k] != 0) {
        TH1D *h1 = (TH1D *)h_eta0[k]->Clone();
        h1->Write();
      }
      if (h_eta1[k] != 0) {
        TH1D *h2 = (TH1D *)h_eta1[k]->Clone();
        h2->Write();
      }
      if (h_eta2[k] != 0) {
        TH1D *h3 = (TH1D *)h_eta2[k]->Clone();
        h3->Write();
      }
      if (h_eta3[k] != 0) {
        TH1D *h4 = (TH1D *)h_eta3[k]->Clone();
        h4->Write();
      }
      if (h_eta4[k] != 0) {
        TH1D *h5 = (TH1D *)h_eta4[k]->Clone();
        h5->Write();
      }
      if (h_dL1[k] != 0) {
        TH1D *h6 = (TH1D *)h_dL1[k]->Clone();
        h6->Write();
      }
      if (h_vtx[k] != 0) {
        TH1D *h7 = (TH1D *)h_vtx[k]->Clone();
        h7->Write();
      }
    }
  }

  if (plotEnergy_) {
    for (int k = 0; k < CalibPlots::npbin; ++k) {
      if (debug)
        std::cout << "Energy[" << k << "] "
                  << " eta " << etalo_ << ":" << etahi_ << ":" << CalibPlots::netabin << " etaEH " << h_etaEH[k].size()
                  << " etaEp " << h_etaEp[k].size() << " etaEE " << h_etaEE[k].size() << " eHcal " << h_eHcal.size()
                  << " mom " << h_mom.size() << " eEcal " << h_eEcal.size() << std::endl;
      for (unsigned int j = 0; j <= (unsigned int)(etahi_ - etalo_); ++j) {
        if (h_etaEH[k].size() > j && h_etaEH[k][j] != nullptr && (all || (k == kp50))) {
          TH1D *hist = (TH1D *)h_etaEH[k][j]->Clone();
          hist->Write();
        }
        if (h_etaEp[k].size() > j && h_etaEp[k][j] != nullptr && (all || (k == kp50))) {
          TH1D *hist = (TH1D *)h_etaEp[k][j]->Clone();
          hist->Write();
        }
        if (h_etaEE[k].size() > j && h_etaEE[k][j] != nullptr && (all || (k == kp50))) {
          TH1D *hist = (TH1D *)h_etaEE[k][j]->Clone();
          hist->Write();
        }
        if (h_etaEE0[k].size() > j && h_etaEE0[k][j] != nullptr && (all || (k == kp50))) {
          TH1D *hist = (TH1D *)h_etaEE0[k][j]->Clone();
          hist->Write();
        }
      }
    }

    for (int j = 0; j < CalibPlots::netabin; ++j) {
      if (h_eHcal.size() > (unsigned int)(j) && (h_eHcal[j] != nullptr)) {
        TH1D *hist = (TH1D *)h_eHcal[j]->Clone();
        hist->Write();
      }
      if (h_mom.size() > (unsigned int)(j) && (h_mom[j] != nullptr)) {
        TH1D *hist = (TH1D *)h_mom[j]->Clone();
        hist->Write();
      }
      if (h_eEcal.size() > (unsigned int)(j) && (h_eEcal[j] != nullptr)) {
        TH1D *hist = (TH1D *)h_eEcal[j]->Clone();
        hist->Write();
      }
    }
  }

  if (plotHists_) {
    if (debug)
      std::cout << "etaE " << h_etaE << std::endl;
    h_etaE->Write();
    for (int i = 0; i < CalibPlots::ndepth; ++i) {
      if (debug)
        std::cout << "Depth[" << i << "] bv " << h_bvlist[i] << ":" << h_bvlist2[i] << ":" << h_bvlist3[i] << " ev "
                  << h_evlist[i] << ":" << h_evlist2[i] << ":" << h_evlist3[i] << std::endl;
      h_bvlist[i]->Write();
      h_bvlist2[i]->Write();
      h_bvlist3[i]->Write();
      h_evlist[i]->Write();
      h_evlist2[i]->Write();
      h_evlist3[i]->Write();
    }
  }
  std::cout << "All done" << std::endl;
  theFile->Close();
}

void CalibPlotProperties::correctEnergy(double &eHcal) {
  double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
  if ((corrPU_ < 0) && (pmom > 0)) {
    double ediff = (t_eHcal30 - t_eHcal10);
    if (t_DetIds1 != 0 && t_DetIds3 != 0) {
      double Etot1(0), Etot3(0);
      // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
      for (unsigned int idet = 0; idet < (*t_DetIds1).size(); idet++) {
        // Apply thresholds if necessary
        bool okcell = (thrForm_ == 0) || ((*t_HitEnergies1)[idet] > threshold((*t_DetIds1)[idet], thrForm_));
        if (okcell) {
          unsigned int id = truncateId((*t_DetIds1)[idet], truncateFlag_, false);
          double cfac = corrFactor_->getCorr(id);
          if ((cFactor_ != 0) && (ifDepth_ != 3) && (ifDepth_ > 0))
            cfac *= cFactor_->getCorr(t_Run, (*t_DetIds1)[idet]);
          if ((cDuplicate_ != nullptr) && (cDuplicate_->doCorr()))
            cfac *= cDuplicate_->getWeight((*t_DetIds1)[idet]);
          double hitEn = cfac * (*t_HitEnergies1)[idet];
          Etot1 += hitEn;
        }
      }
      for (unsigned int idet = 0; idet < (*t_DetIds3).size(); idet++) {
        // Apply thresholds if necessary
        bool okcell = (thrForm_ == 0) || ((*t_HitEnergies3)[idet] > threshold((*t_DetIds3)[idet], thrForm_));
        if (okcell) {
          unsigned int id = truncateId((*t_DetIds3)[idet], truncateFlag_, false);
          double cfac = corrFactor_->getCorr(id);
          if ((cFactor_ != 0) && (ifDepth_ != 3) && (ifDepth_ > 0))
            cfac *= cFactor_->getCorr(t_Run, (*t_DetIds3)[idet]);
          if ((cDuplicate_ != nullptr) && (cDuplicate_->doCorr()))
            cfac *= cDuplicate_->getWeight((*t_DetIds)[idet]);
          double hitEn = cfac * (*t_HitEnergies3)[idet];
          Etot3 += hitEn;
        }
      }
      ediff = (Etot3 - Etot1);
    }
    double fac = puFactor(-corrPU_, t_ieta, pmom, eHcal, ediff);
    eHcal *= fac;
  } else if (corrPU_ > 1) {
    eHcal = puFactorRho(corrPU_, t_ieta, t_rhoh, eHcal);
  }
}

void PlotThisHist(TH1D *hist, const std::string &text, int save) {
  char namep[120];
  sprintf(namep, "c_%s", hist->GetName());
  TCanvas *pad = new TCanvas(namep, namep, 700, 500);
  pad->SetRightMargin(0.10);
  pad->SetTopMargin(0.10);
  hist->GetXaxis()->SetTitleSize(0.04);
  hist->GetYaxis()->SetTitle("Tracks");
  hist->GetYaxis()->SetLabelOffset(0.005);
  hist->GetYaxis()->SetTitleSize(0.04);
  hist->GetYaxis()->SetLabelSize(0.035);
  hist->GetYaxis()->SetTitleOffset(1.10);
  hist->SetMarkerStyle(20);
  hist->SetMarkerColor(2);
  hist->SetLineColor(2);
  hist->Draw("Hist");
  pad->Modified();
  pad->Update();
  TPaveStats *st1 = (TPaveStats *)hist->GetListOfFunctions()->FindObject("stats");
  TPaveText *txt0 = new TPaveText(0.12, 0.91, 0.49, 0.96, "blNDC");
  txt0->SetFillColor(0);
  char txt[100];
  sprintf(txt, "CMS Simulation Preliminary");
  txt0->AddText(txt);
  txt0->Draw("same");
  TPaveText *txt1 = new TPaveText(0.51, 0.91, 0.90, 0.96, "blNDC");
  txt1->SetFillColor(0);
  sprintf(txt, "%s", text.c_str());
  txt1->AddText(txt);
  txt1->Draw("same");
  if (st1 != nullptr) {
    st1->SetY1NDC(0.70);
    st1->SetY2NDC(0.90);
    st1->SetX1NDC(0.65);
    st1->SetX2NDC(0.90);
  }
  pad->Update();
  if (save != 0) {
    if (save > 0)
      sprintf(namep, "%s.pdf", pad->GetName());
    else
      sprintf(namep, "%s.jpg", pad->GetName());
    pad->Print(namep);
  }
}

void PlotHist(const char *hisFileName,
              const std::string &prefix = "",
              const std::string &text = "",
              int flagC = 111,
              int etalo = 0,
              int etahi = 30,
              int save = 0) {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);

  bool plotBasic = (((flagC / 1) % 10) > 0);
  bool plotEnergy = (((flagC / 10) % 10) > 0);
  bool plotHists = (((flagC / 100) % 10) > 0);

  TFile *file = new TFile(hisFileName);
  char name[100], title[200];
  TH1D *hist;
  if ((file != nullptr) && plotBasic) {
    std::cout << "Plot Basic Histos" << std::endl;
    hist = (TH1D *)(file->FindObjectAny("hnvtx"));
    if (hist != nullptr) {
      hist->GetXaxis()->SetTitle("Number of vertices (selected entries)");
      PlotThisHist(hist, text, save);
    }
    hist = (TH1D *)(file->FindObjectAny("hnvtxEv"));
    if (hist != nullptr) {
      hist->GetXaxis()->SetTitle("Number of vertices (selected events)");
      PlotThisHist(hist, text, save);
    }
    hist = (TH1D *)(file->FindObjectAny("hnvtxTk"));
    if (hist != nullptr) {
      hist->GetXaxis()->SetTitle("Number of vertices (selected tracks)");
      PlotThisHist(hist, text, save);
    }
    for (int k = 0; k < CalibPlots::ntitles; ++k) {
      sprintf(name, "%sp%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "Momentum for %s (GeV)", CalibPlots::getTitle(k).c_str());
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%seta%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "#eta for %s", CalibPlots::getTitle(k).c_str());
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
    }
    for (int k = 0; k < CalibPlots::npbin; ++k) {
      sprintf(name, "%seta0%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title,
                "#eta for %s (p = %d:%d GeV)",
                CalibPlots::getTitle(0).c_str(),
                CalibPlots::getP(k),
                CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%seta1%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title,
                "#eta for %s (p = %d:%d GeV)",
                CalibPlots::getTitle(1).c_str(),
                CalibPlots::getP(k),
                CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%seta2%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title,
                "#eta for %s (p = %d:%d GeV)",
                CalibPlots::getTitle(2).c_str(),
                CalibPlots::getP(k),
                CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%seta3%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title,
                "#eta for %s (p = %d:%d GeV)",
                CalibPlots::getTitle(3).c_str(),
                CalibPlots::getP(k),
                CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%seta4%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title,
                "#eta for %s (p = %d:%d GeV)",
                CalibPlots::getTitle(4).c_str(),
                CalibPlots::getP(k),
                CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%sdl1%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "Distance from L1 (p = %d:%d GeV)", CalibPlots::getP(k), CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%svtx%d", prefix.c_str(), k);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "N_{Vertex} (p = %d:%d GeV)", CalibPlots::getP(k), CalibPlots::getP(k + 1));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
    }
  }

  if ((file != nullptr) && plotEnergy) {
    std::cout << "Make plots for good tracks" << std::endl;
    for (int k = 0; k < CalibPlots::npbin; ++k) {
      for (int j = etalo; j <= etahi + 1; ++j) {
        sprintf(name, "%senergyH%d%d", prefix.c_str(), k, j);
        hist = (TH1D *)(file->FindObjectAny(name));
        if (hist != nullptr) {
          if (j > etahi)
            sprintf(title,
                    "HCAL energy for %s (p = %d:%d GeV |#eta| = %d:%d)",
                    CalibPlots::getTitle(3).c_str(),
                    CalibPlots::getP(k),
                    CalibPlots::getP(k + 1),
                    etalo,
                    etahi);
          else
            sprintf(title,
                    "HCAL energy for %s (p = %d:%d GeV |#eta| = %d)",
                    CalibPlots::getTitle(3).c_str(),
                    CalibPlots::getP(k),
                    CalibPlots::getP(k + 1),
                    j);
          hist->GetXaxis()->SetTitle(title);
          PlotThisHist(hist, text, save);
        }
        sprintf(name, "%senergyP%d%d", prefix.c_str(), k, j);
        hist = (TH1D *)(file->FindObjectAny(name));
        if (hist != nullptr) {
          if (j > etahi)
            sprintf(title,
                    "momentum for %s (p = %d:%d GeV |#eta| = %d:%d)",
                    CalibPlots::getTitle(3).c_str(),
                    CalibPlots::getP(k),
                    CalibPlots::getP(k + 1),
                    etalo,
                    etahi);
          else
            sprintf(title,
                    "momentum for %s (p = %d:%d GeV |#eta| = %d)",
                    CalibPlots::getTitle(3).c_str(),
                    CalibPlots::getP(k),
                    CalibPlots::getP(k + 1),
                    j);
          hist->GetXaxis()->SetTitle(title);
          PlotThisHist(hist, text, save);
        }
        sprintf(name, "%senergyE%d%d", prefix.c_str(), k, j);
        hist = (TH1D *)(file->FindObjectAny(name));
        if (hist != nullptr) {
          if (j > etahi)
            sprintf(title,
                    "ECAL energy for %s (p = %d:%d GeV |#eta| = %d:%d)",
                    CalibPlots::getTitle(3).c_str(),
                    CalibPlots::getP(k),
                    CalibPlots::getP(k + 1),
                    etalo,
                    etahi);
          else
            sprintf(title,
                    "ECAL energy for %s (p = %d:%d GeV |#eta| = %d)",
                    CalibPlots::getTitle(3).c_str(),
                    CalibPlots::getP(k),
                    CalibPlots::getP(k + 1),
                    j);
          hist->GetXaxis()->SetTitle(title);
          PlotThisHist(hist, text, save);
        }
        sprintf(name, "%senergyER%d%d", prefix.c_str(), k, j);
        hist = (TH1D *)(file->FindObjectAny(name));
        if (hist != nullptr) {
          std::cout << name << " Mean " << hist->GetMean() << " +- " << hist->GetMeanError() << " Entries "
                    << hist->GetEntries() << " RMS " << hist->GetRMS() << std::endl;
        }
      }
    }

    for (int j = 0; j < CalibPlots::netabin; ++j) {
      sprintf(name, "%senergyH%d", prefix.c_str(), j);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        if (j == 0)
          sprintf(title, "HCAL energy for %s (All)", CalibPlots::getTitle(3).c_str());
        else
          sprintf(title,
                  "HCAL energy for %s (|#eta| = %d:%d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getEta(j - 1),
                  CalibPlots::getEta(j));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%senergyP%d", prefix.c_str(), j);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        if (j == 0)
          sprintf(title, "Track momentum for %s (All)", CalibPlots::getTitle(3).c_str());
        else
          sprintf(title,
                  "Track momentum for %s (|#eta| = %d:%d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getEta(j - 1),
                  CalibPlots::getEta(j));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "%senergyE%d", prefix.c_str(), j);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        if (j == 0)
          sprintf(title, "ECAL energy for %s (All)", CalibPlots::getTitle(3).c_str());
        else
          sprintf(title,
                  "ECAL energy for %s (|#eta| = %d:%d)",
                  CalibPlots::getTitle(3).c_str(),
                  CalibPlots::getEta(j - 1),
                  CalibPlots::getEta(j));
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
    }
  }

  if (plotHists) {
    for (int i = 0; i < CalibPlots::ndepth; i++) {
      sprintf(name, "b_edepth%d", i);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "Total RecHit energy in depth %d (Barrel)", i + 1);
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "b_recedepth%d", i);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "RecHit energy in depth %d (Barrel)", i + 1);
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "b_nrecdepth%d", i);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "#RecHits in depth %d (Barrel)", i + 1);
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "e_edepth%d", i);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "Total RecHit energy in depth %d (Endcap)", i + 1);
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "e_recedepth%d", i);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "RecHit energy in depth %d (Endcap)", i + 1);
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
      sprintf(name, "e_nrecdepth%d", i);
      hist = (TH1D *)(file->FindObjectAny(name));
      if (hist != nullptr) {
        sprintf(title, "#RecHits in depth %d (Endcap)", i + 1);
        hist->GetXaxis()->SetTitle(title);
        PlotThisHist(hist, text, save);
      }
    }
    TH2F *h_etaE = (TH2F *)(file->FindObjectAny("heta"));
    if (h_etaE != nullptr) {
      h_etaE->GetXaxis()->SetTitle("i#eta");
      h_etaE->GetYaxis()->SetTitle("Energy (GeV)");
      char namep[120];
      sprintf(namep, "c_%s", h_etaE->GetName());
      TCanvas *pad = new TCanvas(namep, namep, 700, 700);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      h_etaE->GetXaxis()->SetTitleSize(0.04);
      h_etaE->GetYaxis()->SetLabelOffset(0.005);
      h_etaE->GetYaxis()->SetTitleSize(0.04);
      h_etaE->GetYaxis()->SetLabelSize(0.035);
      h_etaE->GetYaxis()->SetTitleOffset(1.10);
      h_etaE->SetMarkerStyle(20);
      h_etaE->SetMarkerColor(2);
      h_etaE->SetLineColor(2);
      h_etaE->Draw();
      pad->Update();
      TPaveStats *st1 = (TPaveStats *)h_etaE->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        st1->SetY1NDC(0.70);
        st1->SetY2NDC(0.90);
        st1->SetX1NDC(0.65);
        st1->SetX2NDC(0.90);
      }
      if (save != 0) {
        if (save > 0)
          sprintf(namep, "%s.pdf", pad->GetName());
        else
          sprintf(namep, "%s.jpg", pad->GetName());
        pad->Print(namep);
      }
    }
  }
}

class CalibSplit {
public:
  TChain *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t t_Run;
  Int_t t_Event;
  Int_t t_DataType;
  Int_t t_ieta;
  Int_t t_iphi;
  Double_t t_EventWeight;
  Int_t t_nVtx;
  Int_t t_nTrk;
  Int_t t_goodPV;
  Double_t t_l1pt;
  Double_t t_l1eta;
  Double_t t_l1phi;
  Double_t t_l3pt;
  Double_t t_l3eta;
  Double_t t_l3phi;
  Double_t t_p;
  Double_t t_pt;
  Double_t t_phi;
  Double_t t_mindR1;
  Double_t t_mindR2;
  Double_t t_eMipDR;
  Double_t t_eHcal;
  Double_t t_eHcal10;
  Double_t t_eHcal30;
  Double_t t_hmaxNearP;
  Double_t t_rhoh;
  Bool_t t_selectTk;
  Bool_t t_qltyFlag;
  Bool_t t_qltyMissFlag;
  Bool_t t_qltyPVFlag;
  Double_t t_gentrackP;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double> *t_HitEnergies;
  std::vector<bool> *t_trgbits;
  std::vector<unsigned int> *t_DetIds1;
  std::vector<unsigned int> *t_DetIds3;
  std::vector<double> *t_HitEnergies1;
  std::vector<double> *t_HitEnergies3;

  // List of branches
  TBranch *b_t_Run;           //!
  TBranch *b_t_Event;         //!
  TBranch *b_t_DataType;      //!
  TBranch *b_t_ieta;          //!
  TBranch *b_t_iphi;          //!
  TBranch *b_t_EventWeight;   //!
  TBranch *b_t_nVtx;          //!
  TBranch *b_t_nTrk;          //!
  TBranch *b_t_goodPV;        //!
  TBranch *b_t_l1pt;          //!
  TBranch *b_t_l1eta;         //!
  TBranch *b_t_l1phi;         //!
  TBranch *b_t_l3pt;          //!
  TBranch *b_t_l3eta;         //!
  TBranch *b_t_l3phi;         //!
  TBranch *b_t_p;             //!
  TBranch *b_t_pt;            //!
  TBranch *b_t_phi;           //!
  TBranch *b_t_mindR1;        //!
  TBranch *b_t_mindR2;        //!
  TBranch *b_t_eMipDR;        //!
  TBranch *b_t_eHcal;         //!
  TBranch *b_t_eHcal10;       //!
  TBranch *b_t_eHcal30;       //!
  TBranch *b_t_hmaxNearP;     //!
  TBranch *b_t_rhoh;          //!
  TBranch *b_t_selectTk;      //!
  TBranch *b_t_qltyFlag;      //!
  TBranch *b_t_qltyMissFlag;  //!
  TBranch *b_t_qltyPVFlag;    //!
  TBranch *b_t_gentrackP;     //!
  TBranch *b_t_DetIds;        //!
  TBranch *b_t_HitEnergies;   //!
  TBranch *b_t_trgbits;       //!
  TBranch *b_t_DetIds1;       //!
  TBranch *b_t_DetIds3;       //!
  TBranch *b_t_HitEnergies1;  //!
  TBranch *b_t_HitEnergies3;  //!

  // Declaration of output leaf types
  Int_t tout_Run;
  Int_t tout_Event;
  Int_t tout_DataType;
  Int_t tout_ieta;
  Int_t tout_iphi;
  Double_t tout_EventWeight;
  Int_t tout_nVtx;
  Int_t tout_nTrk;
  Int_t tout_goodPV;
  Double_t tout_l1pt;
  Double_t tout_l1eta;
  Double_t tout_l1phi;
  Double_t tout_l3pt;
  Double_t tout_l3eta;
  Double_t tout_l3phi;
  Double_t tout_p;
  Double_t tout_pt;
  Double_t tout_phi;
  Double_t tout_mindR1;
  Double_t tout_mindR2;
  Double_t tout_eMipDR;
  Double_t tout_eHcal;
  Double_t tout_eHcal10;
  Double_t tout_eHcal30;
  Double_t tout_hmaxNearP;
  Double_t tout_rhoh;
  Bool_t tout_selectTk;
  Bool_t tout_qltyFlag;
  Bool_t tout_qltyMissFlag;
  Bool_t tout_qltyPVFlag;
  Double_t tout_gentrackP;
  std::vector<unsigned int> *tout_DetIds;
  std::vector<double> *tout_HitEnergies;
  std::vector<bool> *tout_trgbits;
  std::vector<unsigned int> *tout_DetIds1;
  std::vector<unsigned int> *tout_DetIds3;
  std::vector<double> *tout_HitEnergies1;
  std::vector<double> *tout_HitEnergies3;

  CalibSplit(const char *fname,
             const std::string &dirname,
             const std::string &outFileName,
             double pmin = 40.0,
             double pmax = 60.0,
             bool debug = false);
  virtual ~CalibSplit();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *);
  virtual void Loop(Long64_t nentries = -1);
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  void copyTree();
  void close();

private:
  const std::string fname_, dirnm_, outFileName_;
  const double pmin_, pmax_;
  const bool debug_;
  TFile *outputFile_;
  TDirectoryFile *outputDir_;
  TTree *outputTree_;
};

CalibSplit::CalibSplit(
    const char *fname, const std::string &dirnm, const std::string &outFileName, double pmin, double pmax, bool debug)
    : fname_(fname),
      dirnm_(dirnm),
      outFileName_(outFileName),
      pmin_(pmin),
      pmax_(pmax),
      debug_(debug),
      outputFile_(nullptr),
      outputDir_(nullptr),
      outputTree_(nullptr) {
  char treeName[400];
  sprintf(treeName, "%s/CalibTree", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << " to write tracs of momentum in the range "
            << pmin_ << ":" << pmax_ << " to file " << outFileName_ << std::endl;
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain);
  }
}

CalibSplit::~CalibSplit() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t CalibSplit::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibSplit::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (!fChain->InheritsFrom(TChain::Class()))
    return centry;
  TChain *chain = (TChain *)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void CalibSplit::Init(TChain *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_DetIds = nullptr;
  t_DetIds1 = nullptr;
  t_DetIds3 = nullptr;
  t_HitEnergies = nullptr;
  t_HitEnergies1 = nullptr;
  t_HitEnergies3 = nullptr;
  t_trgbits = nullptr;
  // Set branch addresses and branch pointers
  fChain = tree;
  fCurrent = -1;
  if (!tree)
    return;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_DataType", &t_DataType, &b_t_DataType);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_iphi", &t_iphi, &b_t_iphi);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_nVtx", &t_nVtx, &b_t_nVtx);
  fChain->SetBranchAddress("t_nTrk", &t_nTrk, &b_t_nTrk);
  fChain->SetBranchAddress("t_goodPV", &t_goodPV, &b_t_goodPV);
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_pt", &t_pt, &b_t_pt);
  fChain->SetBranchAddress("t_phi", &t_phi, &b_t_phi);
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_eHcal10", &t_eHcal10, &b_t_eHcal10);
  fChain->SetBranchAddress("t_eHcal30", &t_eHcal30, &b_t_eHcal30);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_rhoh", &t_rhoh, &b_t_rhoh);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyFlag", &t_qltyFlag, &b_t_qltyFlag);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_gentrackP", &t_gentrackP, &b_t_gentrackP);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  fChain->SetBranchAddress("t_trgbits", &t_trgbits, &b_t_trgbits);
  fChain->SetBranchAddress("t_DetIds1", &t_DetIds1, &b_t_DetIds1);
  fChain->SetBranchAddress("t_DetIds3", &t_DetIds3, &b_t_DetIds3);
  fChain->SetBranchAddress("t_HitEnergies1", &t_HitEnergies1, &b_t_HitEnergies1);
  fChain->SetBranchAddress("t_HitEnergies3", &t_HitEnergies3, &b_t_HitEnergies3);
  Notify();

  tout_DetIds = new std::vector<unsigned int>();
  tout_HitEnergies = new std::vector<double>();
  tout_trgbits = new std::vector<bool>();
  tout_DetIds1 = new std::vector<unsigned int>();
  tout_DetIds3 = new std::vector<unsigned int>();
  tout_HitEnergies1 = new std::vector<double>();
  tout_HitEnergies3 = new std::vector<double>();

  outputFile_ = TFile::Open(outFileName_.c_str(), "RECREATE");
  outputFile_->cd();
  outputDir_ = new TDirectoryFile(dirnm_.c_str(), dirnm_.c_str());
  outputDir_->cd();
  outputTree_ = new TTree("CalibTree", "CalibTree");
  outputTree_->Branch("t_Run", &tout_Run);
  outputTree_->Branch("t_Event", &tout_Event);
  outputTree_->Branch("t_DataType", &tout_DataType);
  outputTree_->Branch("t_ieta", &tout_ieta);
  outputTree_->Branch("t_iphi", &tout_iphi);
  outputTree_->Branch("t_EventWeight", &tout_EventWeight);
  outputTree_->Branch("t_nVtx", &tout_nVtx);
  outputTree_->Branch("t_nTrk", &tout_nTrk);
  outputTree_->Branch("t_goodPV", &tout_goodPV);
  outputTree_->Branch("t_l1pt", &tout_l1pt);
  outputTree_->Branch("t_l1eta", &tout_l1eta);
  outputTree_->Branch("t_l1phi", &tout_l1phi);
  outputTree_->Branch("t_l3pt", &tout_l3pt);
  outputTree_->Branch("t_l3eta", &tout_l3eta);
  outputTree_->Branch("t_l3phi", &tout_l3phi);
  outputTree_->Branch("t_p", &tout_p);
  outputTree_->Branch("t_pt", &tout_pt);
  outputTree_->Branch("t_phi", &tout_phi);
  outputTree_->Branch("t_mindR1", &tout_mindR1);
  outputTree_->Branch("t_mindR2", &tout_mindR2);
  outputTree_->Branch("t_eMipDR", &tout_eMipDR);
  outputTree_->Branch("t_eHcal", &tout_eHcal);
  outputTree_->Branch("t_eHcal10", &tout_eHcal10);
  outputTree_->Branch("t_eHcal30", &tout_eHcal30);
  outputTree_->Branch("t_hmaxNearP", &tout_hmaxNearP);
  outputTree_->Branch("t_rhoh", &tout_rhoh);
  outputTree_->Branch("t_selectTk", &tout_selectTk);
  outputTree_->Branch("t_qltyFlag", &tout_qltyFlag);
  outputTree_->Branch("t_qltyMissFlag", &tout_qltyMissFlag);
  outputTree_->Branch("t_qltyPVFlag", &tout_qltyPVFlag);
  outputTree_->Branch("t_gentrackP", &tout_gentrackP);
  outputTree_->Branch("t_DetIds", &tout_DetIds);
  outputTree_->Branch("t_HitEnergies", &tout_HitEnergies);
  outputTree_->Branch("t_trgbits", &tout_trgbits);
  outputTree_->Branch("t_DetIds1", &tout_DetIds1);
  outputTree_->Branch("t_DetIds3", &tout_DetIds3);
  outputTree_->Branch("t_HitEnergies1", &tout_HitEnergies1);
  outputTree_->Branch("t_HitEnergies3", &tout_HitEnergies3);

  std::cout << "Output CalibTree is created in directory " << dirnm_ << " of " << outFileName_ << std::endl;
}

Bool_t CalibSplit::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibSplit::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t CalibSplit::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibSplit::Loop(Long64_t nentries) {
  //   In a ROOT session, you can do:
  //      Root > .L CalibMonitor.C
  //      Root > CalibMonitor t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //   This is the loop skeleton where:
  //      jentry is the global entry number in the chain
  //      ientry is the entry number in the current Tree
  //   Note that the argument to GetEntry must be:
  //      jentry for TChain::GetEntry
  //      ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch

  if (fChain == 0)
    return;

  // Find list of duplicate events
  if (nentries < 0)
    nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << ":" << fChain->GetEntriesFast() << std::endl;
  Long64_t nbytes(0), nb(0);
  unsigned int good(0), reject(0), kount(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (jentry % 1000000 == 0)
      std::cout << "Entry " << jentry << " Run " << t_Run << " Event " << t_Event << std::endl;
    ++kount;
    bool select = ((t_p >= pmin_) && (t_p < pmax_));
    if (!select) {
      ++reject;
      if (debug_)
        std::cout << "Rejected event " << t_Run << " " << t_Event << " " << t_p << std::endl;
      continue;
    }
    ++good;
    copyTree();
    outputTree_->Fill();
  }
  std::cout << "\nSelect " << good << " Reject " << reject << " entries out of a total of " << kount << " counts"
            << std::endl;
  close();
}

void CalibSplit::copyTree() {
  tout_Run = t_Run;
  tout_Event = t_Event;
  tout_DataType = t_DataType;
  tout_ieta = t_ieta;
  tout_iphi = t_iphi;
  tout_EventWeight = t_EventWeight;
  tout_nVtx = t_nVtx;
  tout_nTrk = t_nTrk;
  tout_goodPV = t_goodPV;
  tout_l1pt = t_l1pt;
  tout_l1eta = t_l1eta;
  tout_l1phi = t_l1phi;
  tout_l3pt = t_l3pt;
  tout_l3eta = t_l3eta;
  tout_l3phi = t_l3phi;
  tout_p = t_p;
  tout_pt = t_pt;
  tout_phi = t_phi;
  tout_mindR1 = t_mindR1;
  tout_mindR2 = t_mindR2;
  tout_eMipDR = t_eMipDR;
  tout_eHcal = t_eHcal;
  tout_eHcal10 = t_eHcal10;
  tout_eHcal30 = t_eHcal30;
  tout_hmaxNearP = t_hmaxNearP;
  tout_rhoh = t_rhoh;
  tout_selectTk = t_selectTk;
  tout_qltyFlag = t_qltyFlag;
  tout_qltyMissFlag = t_qltyMissFlag;
  tout_qltyPVFlag = t_qltyPVFlag;
  tout_gentrackP = t_gentrackP;
  tout_DetIds->clear();
  if (t_DetIds != nullptr) {
    tout_DetIds->reserve(t_DetIds->size());
    for (unsigned int i = 0; i < t_DetIds->size(); ++i)
      tout_DetIds->push_back((*t_DetIds)[i]);
  }
  tout_HitEnergies->clear();
  if (t_HitEnergies != nullptr) {
    tout_HitEnergies->reserve(t_HitEnergies->size());
    for (unsigned int i = 0; i < t_HitEnergies->size(); ++i)
      tout_HitEnergies->push_back((*t_HitEnergies)[i]);
  }
  tout_trgbits->clear();
  if (t_trgbits != nullptr) {
    tout_trgbits->reserve(t_trgbits->size());
    for (unsigned int i = 0; i < t_trgbits->size(); ++i)
      tout_trgbits->push_back((*t_trgbits)[i]);
  }
  tout_DetIds1->clear();
  if (t_DetIds1 != nullptr) {
    tout_DetIds1->reserve(t_DetIds1->size());
    for (unsigned int i = 0; i < t_DetIds1->size(); ++i)
      tout_DetIds1->push_back((*t_DetIds1)[i]);
  }
  tout_DetIds3->clear();
  if (t_DetIds3 != nullptr) {
    tout_DetIds3->reserve(t_DetIds3->size());
    for (unsigned int i = 0; i < t_DetIds3->size(); ++i)
      tout_DetIds3->push_back((*t_DetIds3)[i]);
  }
  tout_HitEnergies1->clear();
  if (t_HitEnergies1 != nullptr) {
    tout_HitEnergies1->reserve(t_HitEnergies1->size());
    for (unsigned int i = 0; i < t_HitEnergies1->size(); ++i)
      tout_HitEnergies1->push_back((*t_HitEnergies1)[i]);
  }
  tout_HitEnergies3->clear();
  if (t_HitEnergies1 != nullptr) {
    tout_HitEnergies3->reserve(t_HitEnergies3->size());
    for (unsigned int i = 0; i < t_HitEnergies3->size(); ++i)
      tout_HitEnergies3->push_back((*t_HitEnergies3)[i]);
  }
}

void CalibSplit::close() {
  if (outputFile_) {
    outputDir_->cd();
    std::cout << "file yet to be Written" << std::endl;
    outputTree_->Write();
    std::cout << "file Written" << std::endl;
    outputFile_->Close();
    std::cout << "now doing return" << std::endl;
  }
  outputFile_ = nullptr;
}
