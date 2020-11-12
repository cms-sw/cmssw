//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibMonitor.C+g
//  CalibMonitor c1(fname, dirname, dupFileName, comFileName, outFileName,
//                  prefix, corrFileName, rcorFileName, puCorr, flag, numb,
//                  dataMC, truncateFlag, useGen, scale, useScale, etalo, etahi,
//                  runlo, runhi, phimin, phimax, zside, nvxlo, nvxhi, rbx,
//                  exclude, etamax);
//  c1.Loop();
//  c1.savePlot(histFileName,append,all);
//
//        This will prepare a set of histograms which can be used for a
//        quick fit and display using the methods in CalibFitPlots.C
//
//  GetEntries g1(fname, dirname, dupFileName, bit1, bit2);
//  g1.Loop();
//
//         This looks into the tree *EventInfo* and can provide a set
//         of histograms with event statistics
//
//  CalibPlotProperties cc(fname, dirname, dupFileName, prefix, corrFileName,
//		           rcorFileName, puCorr, flagC, dataMC, truncateFlag,
//                         useGen, scale, useScale, etalo, etahi, runlo, runhi,
//		           phimin, phimax, zside, nvxlo, nvxhi, rbx, exclude,
//                         etamax);
//  cc.Loop();
//  cc.savePlot(histFileName,append,all);
//
//         This makes different style of plots to understand the quality of
//         reconstruction rather than the response plots which are done by
//         CalibMonitor
//
//   where:
//
//   fname   (const char*)     = file name of the input ROOT tree
//                               or name of the file containing a list of
//                               file names of input ROOT trees
//   dirname (std::string)     = name of the directory where Tree resides
//                               (use "HcalIsoTrkAnalyzer")
//   dupFileName (char*)       = name of the file containing list of entries
//                               of duplicate events
//   comFileName (char*)       = name of the file with list of run and event
//                               number to be selected
//   outFileName (char*)       = name of a text file to be created (under
//                               control of value of flag) with information
//                               about events
//   prefix (std::string)      = String to be added to the name of histogram
//                               (usually a 4 character string; default="")
//   corrFileName (char*)      = name of the text file having the correction
//                               factors to be used (default="", no corr.)
//   rcorFileName (char*)      = name of the text file having the correction
//                               factors as a function of run numbers or depth
//                               or entry number to be used for raddam/depth/
//                               pileup dependent correction  (default="",
//                               no corr.)
//   puCorr (int)              = PU correction to be applied or not: 0 no
//                               correction; < 0 use eDelta; > 0 rho dependent
//                               correction (-2)
//   flag (int)                = 7 digit integer (xmlthdo) with control
//                               information (x=3/2/1/0 for having 1000/500/50/
//                               100 bins for response distribution in (0:5);
//                               m=1/0 for (not) making plots for each RBX;
//                               l=3/2/1/0 for type of rcorFileName (3 for
//                               pileup correction using machine learning
//                               method; 2 for overall response corrections;
//                               1 for depth dependence corrections;
//                               0 for raddam corrections);
//                               t=1/0 for applying cut or not on L1 closeness;
//                               h = 0/1/2 for not creating / creating in
//                               recreate mode / creating in append mode
//                               the output text file;
//                               d = 0/1/2/3 produces 3 standard (0,1,2) or
//                               extended (3) set of histograms;
//                               o = 0/1/2 for tight / loose / flexible
//                               selection). Default = 1031
//   flagC (int)               = 6 digit integer (mlthdo) with control
//                               information (m=1/0 for making or not rechit
//                               energy distributions;  l=2/1/0 for type of
//                               correction (2 for overall response corrections;
//                               1 for depth dependence corrections; 0 for
//                               raddam corrections); t=1/0 for applying cut or
//                               not on L1 closeness; h=1/0 for making or not
//                               plots of momentum and total energies in the
//                               two calorimeters ECAL/HCAL; d=1/0 for making
//                               plots of momentum and eta distributions;
//                               o = 0/1/2 for tight / loose / flexible
//                               selection). Default = 1031
//   numb   (int)              = number of eta bins (50 for -25:25)
//   dataMC (bool)             = true/false for data/MC (default true)
//   truncateFlag    (int)     = Flag to treat different depths differently (0)
//                               both depths of ieta 15, 16 of HB as depth 1 (1)
//                               all depths as depth 1 (2), all depths in HE
//                               with values > 1 as depth 2 (3), all depths in
//                               HB with values > 1 as depth 2 (4), all depths
//                               in HB and HE with values > 1 as depth 2 (5)
//                               (default = 0)
//   useGen (bool)             = true/false to use generator level momentum
//                               or reconstruction level momentum
//                               (default = false)
//   scale (double)            = energy scale if correction factor to be used
//                               (default = 1.0)
//   useScale (int)            = application of scale factor (0: nowehere,
//                               1: barrel; 2: endcap, 3: everywhere)
//                               barrel => |ieta| < 16; endcap => |ieta| > 15
//                               (default = 0)
//   etalo/etahi (int,int)     = |eta| ranges (default = 0:30)
//   runlo  (int)              = lower value of run number to be included (+ve)
//                               or excluded (-ve) (default = 0)
//   runhi  (int)              = higher value of run number to be included
//                               (+ve) or excluded (-ve) (default = 9999999)
//   phimin          (int)     = minimum iphi value (default = 1)
//   phimax          (int)     = maximum iphi value (default = 72)
//   zside           (int)     = the side of the detector if phimin and phimax
//                               differ from 1-72 (default = 1)
//   nvxlo           (int)     = minimum # of vertex in event to be used
//                               (default = 0)
//   nvxhi           (int)     = maximum # of vertex in event to be used
//                               (default = 1000)
//   rbx             (int)     = zside*(Subdet*100+RBX #) to be consdered
//                               (default = 0). For HEP17 it will be 217
//   exclude         (bool)    = RBX specified by *rbx* to be exluded or only
//                               considered (default = false)
//   etamax          (bool)    = if set and if the corr-factor not found in the
//                               corrFactor table, the corr-factor for the
//                               corresponding zside, depth=1 and maximum ieta
//                               in the table is taken (default = false)
//
//   histFileName (std::string)= name of the file containing saved histograms
//   append (bool)             = true/false if the histogram file to be opened
//                               in append/output mode (default = true)
//   all (bool)                = true/false if all histograms to be saved or
//                               not (default = false)
//
//   bitX (unsigned int)       = bit number of the HLT used in the selection
//        (X = 1, 2)             for example the bits of HLT_IsoTrackHB(HE)
//////////////////////////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
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
#include <map>
#include <vector>
#include <string>

#include "CalibCorr.C"

class CalibMonitor {
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

  struct counter {
    static const int npsize = 4;
    counter() {
      total = 0;
      for (int k = 0; k < npsize; ++k)
        count[k] = 0;
    };
    unsigned int total, count[npsize];
  };

  CalibMonitor(const char *fname,
               const std::string &dirname,
               const char *dupFileName,
               const char *comFileName,
               const char *outFileName,
               const std::string &prefix = "",
               const char *corrFileName = "",
               const char *rcorFileName = "",
               int puCorr = -2,
               int flag = 1031,
               int numb = 50,
               bool datMC = true,
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
               int rbx = 0,
               bool exclude = false,
               bool etamax = false);
  virtual ~CalibMonitor();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *, const char *, const char *, const char *);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  bool goodTrack(double &eHcal, double &cut, const Long64_t &entry, bool debug);
  bool selectPhi(bool debug);
  void plotHist(int type, int num, bool save = false);
  template <class Hist>
  void drawHist(Hist *, TCanvas *);
  void savePlot(const std::string &theName, bool append = true, bool all = false);
  void correctEnergy(double &ener, const Long64_t &entry);

private:
  static const unsigned int npbin = 6, kp50 = 3;
  CalibCorrFactor *corrFactor_;
  CalibCorr *cFactor_;
  CalibSelectRBX *cSelect_;
  const std::string fname_, dirnm_, prefix_, outFileName_;
  const int corrPU_, flag_, numb_;
  const bool dataMC_, useGen_;
  const int truncateFlag_;
  const int etalo_, etahi_;
  int runlo_, runhi_;
  const int phimin_, phimax_, zside_, nvxlo_, nvxhi_, rbx_;
  bool exclude_, corrE_, cutL1T_, selRBX_;
  bool includeRun_;
  int coarseBin_, etamp_, etamn_, plotType_;
  int flexibleSelect_, ifDepth_;
  double log2by18_;
  std::ofstream fileout_;
  std::vector<Long64_t> entries_;
  std::vector<std::pair<int, int> > events_;
  std::vector<double> etas_, ps_, dl1_;
  std::vector<int> nvx_, ietas_;
  std::vector<TH1D *> h_rbx, h_etaF[npbin], h_etaB[npbin];
  std::vector<TProfile *> h_etaX[npbin];
  std::vector<TH1D *> h_etaR[npbin], h_nvxR[npbin], h_dL1R[npbin];
  std::vector<TH1D *> h_pp[npbin];
};

CalibMonitor::CalibMonitor(const char *fname,
                           const std::string &dirnm,
                           const char *dupFileName,
                           const char *comFileName,
                           const char *outFName,
                           const std::string &prefix,
                           const char *corrFileName,
                           const char *rcorFileName,
                           int puCorr,
                           int flag,
                           int numb,
                           bool dataMC,
                           int truncate,
                           bool useGen,
                           double scale,
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
                           int rbx,
                           bool exc,
                           bool etam)
    : corrFactor_(nullptr),
      cFactor_(nullptr),
      cSelect_(nullptr),
      fname_(fname),
      dirnm_(dirnm),
      prefix_(prefix),
      outFileName_(std::string(outFName)),
      corrPU_(puCorr),
      flag_(flag),
      numb_(numb),
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
      rbx_(rbx),
      exclude_(exc),
      includeRun_(true) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree

  plotType_ = ((flag_ / 10) % 10);
  if (plotType_ < 0 || plotType_ > 3)
    plotType_ = 3;
  flexibleSelect_ = (((flag_ / 1) % 10));
  cutL1T_ = ((flag_ / 1000) % 10);
  ifDepth_ = ((flag_ / 10000) % 10);
  selRBX_ = (((flag_ / 100000) % 10) > 0);
  coarseBin_ = ((flag_ / 1000000) % 10);
  log2by18_ = std::log(2.5) / 18.0;
  if (runlo_ < 0 || runhi_ < 0) {
    runlo_ = std::abs(runlo_);
    runhi_ = std::abs(runhi_);
    includeRun_ = false;
  }
  char treeName[400];
  sprintf(treeName, "%s/CalibTree", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << " flags " << flexibleSelect_ << "|"
            << plotType_ << "|" << corrPU_ << "\n cons " << log2by18_ << " eta range " << etalo_ << ":" << etahi_
            << " run range " << runlo_ << ":" << runhi_ << " (inclusion flag " << includeRun_ << ")\n Selection of RBX "
            << selRBX_ << " Vertex Range " << nvxlo_ << ":" << nvxhi_ << "\n corrFileName: " << corrFileName
            << " useScale " << useScale << ":" << scale << ":" << etam << "\n rcorFileName: " << rcorFileName
            << " flag " << ifDepth_ << std::endl;
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    corrFactor_ = new CalibCorrFactor(corrFileName, useScale, scale, etam, false);
    Init(chain, dupFileName, comFileName, outFName);
    if (std::string(rcorFileName) != "") {
      cFactor_ = new CalibCorr(rcorFileName, ifDepth_, false);
    } else {
      ifDepth_ = 0;
    }
    if (rbx != 0)
      cSelect_ = new CalibSelectRBX(rbx, false);
  }
}

CalibMonitor::~CalibMonitor() {
  delete corrFactor_;
  delete cFactor_;
  delete cSelect_;
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t CalibMonitor::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibMonitor::LoadTree(Long64_t entry) {
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

void CalibMonitor::Init(TChain *tree, const char *dupFileName, const char *comFileName, const char *outFileName) {
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

  if (strcmp(dupFileName, "") != 0) {
    ifstream infil1(dupFileName);
    if (!infil1.is_open()) {
      std::cout << "Cannot open duplicate file " << dupFileName << std::endl;
    } else {
      while (1) {
        Long64_t jentry;
        infil1 >> jentry;
        if (!infil1.good())
          break;
        entries_.push_back(jentry);
      }
      infil1.close();
      std::cout << "Reads a list of " << entries_.size() << " events from " << dupFileName << std::endl;
    }
  } else {
    std::cout << "No duplicate events in the input file" << std::endl;
  }

  if (strcmp(comFileName, "") != 0) {
    ifstream infil2(comFileName);
    if (!infil2.is_open()) {
      std::cout << "Cannot open selection file " << comFileName << std::endl;
    } else {
      while (1) {
        int irun, ievt;
        infil2 >> irun >> ievt;
        if (!infil2.good())
          break;
        std::pair<int, int> key(irun, ievt);
        events_.push_back(key);
      }
      infil2.close();
      std::cout << "Reads a list of " << events_.size() << " events from " << comFileName << std::endl;
    }
  } else {
    std::cout << "No event list provided for selection" << std::endl;
  }

  if (((flag_ / 100) % 10) > 0) {
    if (((flag_ / 100) % 10) == 2) {
      fileout_.open(outFileName, std::ofstream::out);
      std::cout << "Opens " << outFileName << " in output mode" << std::endl;
    } else {
      fileout_.open(outFileName, std::ofstream::app);
      std::cout << "Opens " << outFileName << " in append mode" << std::endl;
    }
    fileout_ << "Input file: " << fname_ << " Directory: " << dirnm_ << " Prefix: " << prefix_ << std::endl;
  }

  double xbins[99];
  int nbins(-1);
  if (plotType_ == 0) {
    double xbin[9] = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
    for (int i = 0; i < 9; ++i) {
      etas_.push_back(xbin[i]);
      xbins[i] = xbin[i];
    }
    nbins = 8;
  } else if (plotType_ == 1) {
    double xbin[11] = {-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0};
    for (int i = 0; i < 11; ++i) {
      etas_.push_back(xbin[i]);
      xbins[i] = xbin[i];
    }
    nbins = 10;
  } else if (plotType_ == 2) {
    double xbin[23] = {-23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, 0.0,
                       3.0,   5.0,   7.0,   9.0,   11.0,  13.0,  15.0,  17.0, 19.0, 21.0, 23.0};
    for (int i = 0; i < 23; ++i) {
      etas_.push_back(xbin[i]);
      xbins[i] = xbin[i];
    }
    nbins = 22;
  } else {
    double xbina[99];
    int neta = numb_ / 2;
    for (int k = 0; k < neta; ++k) {
      xbina[k] = (k - neta) - 0.5;
      xbina[numb_ - k] = (neta - k) + 0.5;
    }
    xbina[neta] = 0;
    for (int i = 0; i < numb_ + 1; ++i) {
      etas_.push_back(xbina[i]);
      xbins[i] = xbina[i];
      ++nbins;
    }
  }
  int ipbin[npbin] = {10, 20, 30, 40, 60, 100};
  for (unsigned int i = 0; i < npbin; ++i)
    ps_.push_back((double)(ipbin[i]));
  int npvtx[6] = {0, 7, 10, 13, 16, 100};
  for (int i = 0; i < 6; ++i)
    nvx_.push_back(npvtx[i]);
  double dl1s[9] = {0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  int ietas[4] = {0, 13, 18, 23};
  for (int i = 0; i < 4; ++i)
    ietas_.push_back(ietas[i]);
  int nxbin(100);
  double xlow(0.25), xhigh(5.25);
  if (coarseBin_ == 1) {
    nxbin = 50;
  } else if (coarseBin_ > 1) {
    xlow = 0.0;
    xhigh = 5.0;
    if (coarseBin_ == 2)
      nxbin = 500;
    else
      nxbin = 1000;
  }

  char name[20], title[200];
  std::string titl[5] = {
      "All tracks", "Good quality tracks", "Selected tracks", "Tracks with charge isolation", "Tracks MIP in ECAL"};
  for (int i = 0; i < 9; ++i)
    dl1_.push_back(dl1s[i]);
  unsigned int kp = (ps_.size() - 1);
  for (unsigned int k = 0; k < kp; ++k) {
    for (unsigned int j = 0; j <= ietas_.size(); ++j) {
      sprintf(name, "%spp%d%d", prefix_.c_str(), k, j);
      if (j == 0)
        sprintf(title, "E/p for %s (p = %d:%d GeV All)", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
      else if (j == ietas_.size())
        sprintf(title,
                "E/p for %s (p = %d:%d GeV |#eta| %d:%d)",
                titl[4].c_str(),
                ipbin[k],
                ipbin[k + 1],
                ietas_[0],
                ietas_[j - 1]);
      else
        sprintf(title,
                "E/p for %s (p = %d:%d GeV |#eta| %d:%d)",
                titl[4].c_str(),
                ipbin[k],
                ipbin[k + 1],
                ietas_[j - 1],
                ietas_[j]);
      h_pp[k].push_back(new TH1D(name, title, 100, 10.0, 110.0));
      int kk = h_pp[k].size() - 1;
      h_pp[k][kk]->Sumw2();
    }
  }
  if (plotType_ <= 1) {
    std::cout << "Book Histos for Standard" << std::endl;
    for (unsigned int k = 0; k < kp; ++k) {
      for (unsigned int i = 0; i < nvx_.size(); ++i) {
        sprintf(name, "%setaX%d%d", prefix_.c_str(), k, i);
        if (i == 0) {
          sprintf(title, "%s (p = %d:%d GeV all vertices)", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
        } else {
          sprintf(
              title, "%s (p = %d:%d GeV # Vtx %d:%d)", titl[4].c_str(), ipbin[k], ipbin[k + 1], nvx_[i - 1], nvx_[i]);
        }
        h_etaX[k].push_back(new TProfile(name, title, nbins, xbins));
        unsigned int kk = h_etaX[k].size() - 1;
        h_etaX[k][kk]->Sumw2();
        sprintf(name, "%snvxR%d%d", prefix_.c_str(), k, i);
        if (i == 0) {
          sprintf(title, "E/p for %s (p = %d:%d GeV all vertices)", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
        } else {
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV # Vtx %d:%d)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  nvx_[i - 1],
                  nvx_[i]);
        }
        h_nvxR[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        kk = h_nvxR[k].size() - 1;
        h_nvxR[k][kk]->Sumw2();
      }
      for (unsigned int j = 0; j < etas_.size(); ++j) {
        sprintf(name, "%sratio%d%d", prefix_.c_str(), k, j);
        if (j == 0) {
          sprintf(title, "E/p for %s (p = %d:%d GeV)", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
        } else {
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV #eta %4.1f:%4.1f)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  etas_[j - 1],
                  etas_[j]);
        }
        h_etaF[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        unsigned int kk = h_etaF[k].size() - 1;
        h_etaF[k][kk]->Sumw2();
        sprintf(name, "%setaR%d%d", prefix_.c_str(), k, j);
        h_etaR[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        kk = h_etaR[k].size() - 1;
        h_etaR[k][kk]->Sumw2();
      }
      for (unsigned int j = 1; j <= ietas_.size(); ++j) {
        sprintf(name, "%setaB%d%d", prefix_.c_str(), k, j);
        if (j == ietas_.size())
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV |#eta| %d:%d)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  ietas_[0],
                  ietas_[j - 1]);
        else
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV |#eta| %d:%d)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  ietas_[j - 1],
                  ietas_[j]);
        h_etaB[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        unsigned int kk = h_etaB[k].size() - 1;
        h_etaB[k][kk]->Sumw2();
      }
      for (unsigned int j = 0; j < dl1_.size(); ++j) {
        sprintf(name, "%sdl1R%d%d", prefix_.c_str(), k, j);
        if (j == 0) {
          sprintf(title, "E/p for %s (p = %d:%d GeV All d_{L1})", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
        } else {
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV d_{L1} %4.2f:%4.2f)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  dl1_[j - 1],
                  dl1_[j]);
        }
        h_dL1R[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        unsigned int kk = h_dL1R[k].size() - 1;
        h_dL1R[k][kk]->Sumw2();
      }
    }
    for (unsigned int i = 0; i < nvx_.size(); ++i) {
      sprintf(name, "%setaX%d%d", prefix_.c_str(), kp, i);
      if (i == 0) {
        sprintf(title, "%s (All Momentum all vertices)", titl[4].c_str());
      } else {
        sprintf(title, "%s (All Momentum # Vtx %d:%d)", titl[4].c_str(), nvx_[i - 1], nvx_[i]);
      }
      h_etaX[npbin - 1].push_back(new TProfile(name, title, nbins, xbins));
      unsigned int kk = h_etaX[npbin - 1].size() - 1;
      h_etaX[npbin - 1][kk]->Sumw2();
      sprintf(name, "%snvxR%d%d", prefix_.c_str(), kp, i);
      if (i == 0) {
        sprintf(title, "E/p for %s (All Momentum all vertices)", titl[4].c_str());
      } else {
        sprintf(title, "E/p for %s (All Momentum # Vtx %d:%d)", titl[4].c_str(), nvx_[i - 1], nvx_[i]);
      }
      h_nvxR[npbin - 1].push_back(new TH1D(name, title, 200, 0., 10.));
      kk = h_nvxR[npbin - 1].size() - 1;
      h_nvxR[npbin - 1][kk]->Sumw2();
    }
    for (unsigned int j = 0; j < etas_.size(); ++j) {
      sprintf(name, "%sratio%d%d", prefix_.c_str(), kp, j);
      if (j == 0) {
        sprintf(title, "E/p for %s (All momentum)", titl[4].c_str());
      } else {
        sprintf(title, "E/p for %s (All momentum #eta %4.1f:%4.1f)", titl[4].c_str(), etas_[j - 1], etas_[j]);
      }
      h_etaF[npbin - 1].push_back(new TH1D(name, title, 200, 0., 10.));
      unsigned int kk = h_etaF[npbin - 1].size() - 1;
      h_etaF[npbin - 1][kk]->Sumw2();
      sprintf(name, "%setaR%d%d", prefix_.c_str(), kp, j);
      h_etaR[npbin - 1].push_back(new TH1D(name, title, 200, 0., 10.));
      kk = h_etaR[npbin - 1].size() - 1;
      h_etaR[npbin - 1][kk]->Sumw2();
    }
    for (unsigned int j = 1; j <= ietas_.size(); ++j) {
      sprintf(name, "%setaB%d%d", prefix_.c_str(), kp, j);
      if (j == ietas_.size())
        sprintf(title, "E/p for %s (All momentum |#eta| %d:%d)", titl[4].c_str(), ietas_[0], ietas_[j - 1]);
      else
        sprintf(title, "E/p for %s (All momentum |#eta| %d:%d)", titl[4].c_str(), ietas_[j - 1], ietas_[j]);
      h_etaB[npbin - 1].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
      unsigned int kk = h_etaB[npbin - 1].size() - 1;
      h_etaB[npbin - 1][kk]->Sumw2();
    }
    for (unsigned int j = 0; j < dl1_.size(); ++j) {
      sprintf(name, "%sdl1R%d%d", prefix_.c_str(), kp, j);
      if (j == 0) {
        sprintf(title, "E/p for %s (All momentum)", titl[4].c_str());
      } else {
        sprintf(title, "E/p for %s (All momentum d_{L1} %4.2f:%4.2f)", titl[4].c_str(), dl1_[j - 1], dl1_[j]);
      }
      h_dL1R[npbin - 1].push_back(new TH1D(name, title, 200, 0., 10.));
      unsigned int kk = h_dL1R[npbin - 1].size() - 1;
      h_dL1R[npbin - 1][kk]->Sumw2();
    }
  } else {
    std::cout << "Book Histos for Non-Standard " << etas_.size() << ":" << kp50 << std::endl;
    unsigned int kp = (ps_.size() - 1);
    for (unsigned int k = 0; k < kp; ++k) {
      for (unsigned int j = 0; j < etas_.size(); ++j) {
        sprintf(name, "%sratio%d%d", prefix_.c_str(), k, j);
        if (j == 0) {
          sprintf(title, "E/p for %s (p = %d:%d GeV)", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
        } else {
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV #eta %4.1f:%4.1f)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  etas_[j - 1],
                  etas_[j]);
        }
        h_etaF[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        unsigned int kk = h_etaF[k].size() - 1;
        h_etaF[k][kk]->Sumw2();
      }
      for (unsigned int j = 1; j <= ietas_.size(); ++j) {
        sprintf(name, "%setaB%d%d", prefix_.c_str(), k, j);
        if (j == ietas_.size())
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV |#eta| %d:%d)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  ietas_[0],
                  ietas_[j - 1]);
        else
          sprintf(title,
                  "E/p for %s (p = %d:%d GeV |#eta| %d:%d)",
                  titl[4].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  ietas_[j - 1],
                  ietas_[j]);
        h_etaB[k].push_back(new TH1D(name, title, nxbin, xlow, xhigh));
        unsigned int kk = h_etaB[k].size() - 1;
        h_etaB[k][kk]->Sumw2();
      }
    }
  }
  if (selRBX_) {
    for (unsigned int j = 1; j <= 18; ++j) {
      sprintf(name, "%sRBX%d%d", prefix_.c_str(), kp50, j);
      sprintf(title, "E/p for RBX%d (p = %d:%d GeV |#eta| %d:%d)", j, ipbin[kp50], ipbin[kp50 + 1], etalo_, etahi_);
      h_rbx.push_back(new TH1D(name, title, nxbin, xlow, xhigh));
      h_rbx[j - 1]->Sumw2();
    }
  }
}

Bool_t CalibMonitor::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibMonitor::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t CalibMonitor::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibMonitor::Loop() {
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
  std::map<int, counter> runSum, runEn1, runEn2;

  // Find list of duplicate events
  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << std::endl;
  Long64_t nbytes(0), nb(0);
  unsigned int duplicate(0), good(0), kount(0);
  unsigned int kp1 = ps_.size() - 1;
  unsigned int kv1 = 0;
  std::vector<int> kounts(kp1, 0);
  std::vector<int> kount50(20, 0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    //for (Long64_t jentry=0; jentry<200;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (jentry % 100000 == 0)
      std::cout << "Entry " << jentry << " Run " << t_Run << " Event " << t_Event << std::endl;
    double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
    bool p4060 = ((pmom >= 40.0) && (pmom <= 60.0));
    if (p4060)
      ++kount50[0];
    bool select = (std::find(entries_.begin(), entries_.end(), jentry) == entries_.end());
    if (!select) {
      ++duplicate;
      if (debug)
        std::cout << "Duplicate event " << t_Run << " " << t_Event << " " << t_p << std::endl;
      continue;
    }
    if (p4060)
      ++kount50[1];
    bool selRun = (includeRun_ ? ((t_Run >= runlo_) && (t_Run <= runhi_)) : ((t_Run < runlo_) || (t_Run > runhi_)));
    if (select && p4060)
      ++kount50[2];
    select =
        (selRun && (fabs(t_ieta) >= etalo_) && (fabs(t_ieta) <= etahi_) && (t_nVtx >= nvxlo_) && (t_nVtx <= nvxhi_));
    if (select && p4060)
      ++kount50[3];
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
    if (p4060)
      ++kount50[4];
    select = (!cutL1T_ || (t_mindR1 >= 0.5));
    if (!select) {
      if (debug)
        std::cout << "Reject Run # " << t_Run << " Event # " << t_Event << " too close to L1 trigger " << t_mindR1
                  << std::endl;
      continue;
    }
    if (p4060)
      ++kount50[5];
    select = ((events_.size() == 0) ||
              (std::find(events_.begin(), events_.end(), std::pair<int, int>(t_Run, t_Event)) != events_.end()));
    if (!select) {
      if (debug)
        std::cout << "Reject Run # " << t_Run << " Event # " << t_Event << " not in the selection list" << std::endl;
      continue;
    }
    if (p4060)
      ++kount50[6];
    if ((ifDepth_ == 3) && (cFactor_ != nullptr) && (cFactor_->absent(ientry)))
      continue;

    // if (Cut(ientry) < 0) continue;
    int kp(-1), jp(-1), jp1(-1);
    for (unsigned int k = 1; k < ps_.size(); ++k) {
      if (pmom >= ps_[k - 1] && pmom < ps_[k]) {
        kp = k - 1;
        break;
      }
    }
    unsigned int kv = nvx_.size() - 1;
    for (unsigned int k = 1; k < nvx_.size(); ++k) {
      if (t_goodPV >= nvx_[k - 1] && t_goodPV < nvx_[k]) {
        kv = k;
        break;
      }
    }
    unsigned int kd1 = 0;
    unsigned int kd = dl1_.size() - 1;
    for (unsigned int k = 1; k < dl1_.size(); ++k) {
      if (t_mindR1 >= dl1_[k - 1] && t_mindR1 < dl1_[k]) {
        kd = k;
        break;
      }
    }
    double eta = (t_ieta > 0) ? ((double)(t_ieta)-0.001) : ((double)(t_ieta) + 0.001);
    for (unsigned int j = 1; j < etas_.size(); ++j) {
      if (eta > etas_[j - 1] && eta < etas_[j]) {
        jp = j;
        break;
      }
    }
    for (unsigned int j = 1; j < ietas_.size(); ++j) {
      if (std::abs(t_ieta) > ietas_[j - 1] && std::abs(t_ieta) <= ietas_[j]) {
        jp1 = j - 1;
        break;
      }
    }
    int jp2 = (jp1 >= 0) ? (int)(ietas_.size() - 1) : jp1;
    if (debug)
      std::cout << "Bin " << kp << ":" << kp1 << ":" << kv << ":" << kv1 << ":" << kd << ":" << kd1 << ":" << jp << ":"
                << jp1 << ":" << jp2 << std::endl;
    double cut = (pmom > 20) ? ((flexibleSelect_ == 0) ? 2.0 : 10.0) : 0.0;
    //  double rcut= (pmom > 20) ? 0.25: 0.1;
    double rcut(-1000.0);

    // Selection of good track and energy measured in Hcal
    double rat(1.0), eHcal(t_eHcal);
    if (corrFactor_->doCorr() || (cFactor_ != nullptr)) {
      eHcal = 0;
      for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
        // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
        double cfac(1.0);
        if (corrFactor_->doCorr()) {
          unsigned int id = truncateId((*t_DetIds)[k], truncateFlag_, false);
          cfac = corrFactor_->getCorr(id);
        }
        if ((cFactor_ != nullptr) && (ifDepth_ != 3))
          cfac *= cFactor_->getCorr(t_Run, (*t_DetIds)[k]);
        eHcal += (cfac * ((*t_HitEnergies)[k]));
        if (debug) {
          int subdet, zside, ieta, iphi, depth;
          unpackDetId((*t_DetIds)[k], subdet, zside, ieta, iphi, depth);
          std::cout << zside << ":" << ieta << ":" << depth << " Corr " << cfac << " " << (*t_HitEnergies)[k] << " Out "
                    << eHcal << std::endl;
        }
      }
    }
    bool goodTk = goodTrack(eHcal, cut, jentry, debug);
    bool selPhi = selectPhi(debug);
    if (p4060) {
      if (t_qltyFlag) {
        ++kount50[7];
        if (t_selectTk) {
          ++kount50[8];
          if (t_hmaxNearP < cut) {
            ++kount50[9];
            if (t_eMipDR < 1.0) {
              ++kount50[10];
              if (eHcal > 0.001) {
                ++kount50[11];
                if (selPhi)
                  ++kount50[12];
              }
            }
          }
        }
      }
    }
    if (pmom > 0)
      rat = (eHcal / (pmom - t_eMipDR));
    if (debug) {
      std::cout << "Entry " << jentry << " p|eHcal|ratio " << pmom << "|" << t_eHcal << "|" << eHcal << "|" << rat
                << "|" << kp << "|" << kv << "|" << jp << " Cuts " << t_qltyFlag << "|" << t_selectTk << "|"
                << (t_hmaxNearP < cut) << "|" << (t_eMipDR < 1.0) << "|" << goodTk << "|" << (rat > rcut)
                << " Select Phi " << selPhi << std::endl;
      std::cout << "D1 : " << kp << ":" << kp1 << ":" << kv << ":" << kv1 << ":" << kd << ":" << kd1 << ":" << jp
                << std::endl;
    }
    if (goodTk && (kp >= 0) && selPhi) {
      if (p4060)
        ++kount50[13];
      if (t_eHcal < 0.01) {
        std::map<int, counter>::const_iterator itr = runEn1.find(t_Run);
        if (itr == runEn1.end()) {
          counter knt;
          if ((kp >= 0) && (kp < counter::npsize))
            knt.count[kp] = 1;
          knt.total = 1;
          runEn1[t_Run] = knt;
        } else {
          counter knt = runEn1[t_Run];
          if ((kp >= 0) && (kp < counter::npsize))
            ++knt.count[kp];
          ++knt.total;
          runEn1[t_Run] = knt;
        }
      }
      if (t_eMipDR < 0.01 && t_eHcal < 0.01) {
        if (p4060)
          ++kount50[14];
        std::map<int, counter>::const_iterator itr = runEn2.find(t_Run);
        if (itr == runEn2.end()) {
          counter knt;
          if ((kp >= 0) && (kp < counter::npsize))
            knt.count[kp] = 1;
          knt.total = 1;
          runEn2[t_Run] = knt;
        } else {
          counter knt = runEn2[t_Run];
          if ((kp >= 0) && (kp < counter::npsize))
            ++knt.count[kp];
          ++knt.total;
          runEn2[t_Run] = knt;
        }
      }
      if (rat > rcut) {
        if (p4060)
          ++kount50[15];
        if (plotType_ <= 1) {
          h_etaX[kp][kv]->Fill(eta, rat, t_EventWeight);
          h_etaX[kp][kv1]->Fill(eta, rat, t_EventWeight);
          h_nvxR[kp][kv]->Fill(rat, t_EventWeight);
          h_nvxR[kp][kv1]->Fill(rat, t_EventWeight);
          h_dL1R[kp][kd]->Fill(rat, t_EventWeight);
          h_dL1R[kp][kd1]->Fill(rat, t_EventWeight);
          if (jp > 0)
            h_etaR[kp][jp]->Fill(rat, t_EventWeight);
          h_etaR[kp][0]->Fill(rat, t_EventWeight);
        }
        h_pp[kp][0]->Fill(pmom, t_EventWeight);
        if (jp1 >= 0) {
          h_pp[kp][jp1 + 1]->Fill(pmom, t_EventWeight);
          h_pp[kp][jp2 + 1]->Fill(pmom, t_EventWeight);
        }
        if (kp == (int)(kp50)) {
          std::map<int, counter>::const_iterator itr = runSum.find(t_Run);
          if (itr == runSum.end()) {
            counter knt;
            if ((kp >= 0) && (kp < counter::npsize))
              knt.count[kp] = 1;
            knt.total = 1;
            runSum[t_Run] = knt;
          } else {
            counter knt = runSum[t_Run];
            if ((kp >= 0) && (kp < counter::npsize))
              ++knt.count[kp];
            ++knt.total;
            runSum[t_Run] = knt;
          }
        }
        if ((!dataMC_) || (t_mindR1 > 0.5) || (t_DataType == 1)) {
          if (p4060)
            ++kount50[16];
          ++kounts[kp];
          if (plotType_ <= 1) {
            if (jp > 0)
              h_etaF[kp][jp]->Fill(rat, t_EventWeight);
            h_etaF[kp][0]->Fill(rat, t_EventWeight);
          } else {
            if (debug) {
              std::cout << "kp " << kp << h_etaF[kp].size() << std::endl;
            }
            if (jp > 0)
              h_etaF[kp][jp]->Fill(rat, t_EventWeight);
            h_etaF[kp][0]->Fill(rat, t_EventWeight);
            if (jp1 >= 0) {
              h_etaB[kp][jp1]->Fill(rat, t_EventWeight);
              h_etaB[kp][jp2]->Fill(rat, t_EventWeight);
            }
          }
          if (selRBX_ && (kp == (int)(kp50)) && ((t_ieta * zside_) > 0)) {
            int rbx = (t_iphi > 70) ? 0 : (t_iphi + 1) / 4;
            h_rbx[rbx]->Fill(rat, t_EventWeight);
          }
        }
        if (pmom > 20.0) {
          if (plotType_ <= 1) {
            h_etaX[kp1][kv]->Fill(eta, rat, t_EventWeight);
            h_etaX[kp1][kv1]->Fill(eta, rat, t_EventWeight);
            h_nvxR[kp1][kv]->Fill(rat, t_EventWeight);
            h_nvxR[kp1][kv1]->Fill(rat, t_EventWeight);
            h_dL1R[kp1][kd]->Fill(rat, t_EventWeight);
            h_dL1R[kp1][kd1]->Fill(rat, t_EventWeight);
            if (jp > 0)
              h_etaR[kp1][jp]->Fill(rat, t_EventWeight);
            h_etaR[kp1][0]->Fill(rat, t_EventWeight);
            if (jp1 >= 0) {
              h_etaB[kp][jp1]->Fill(rat, t_EventWeight);
              h_etaB[kp][jp2]->Fill(rat, t_EventWeight);
            }
          }
          if (p4060)
            ++kount50[17];
        }
      }
    }
    if (pmom > 20.0) {
      kount++;
      if (((flag_ / 100) % 10) != 0) {
        good++;
        fileout_ << good << " " << jentry << " " << t_Run << " " << t_Event << " " << t_ieta << " " << pmom
                 << std::endl;
      }
    }
  }
  unsigned int k(0);
  std::cout << "\nSummary of entries with " << runSum.size() << " runs\n";
  for (std::map<int, counter>::iterator itr = runSum.begin(); itr != runSum.end(); ++itr, ++k)
    std::cout << "[" << k << "] Run " << itr->first << " Total " << (itr->second).total << " in p-bins "
              << (itr->second).count[0] << ":" << (itr->second).count[1] << ":" << (itr->second).count[2] << ":"
              << (itr->second).count[3] << std::endl;
  k = 0;
  std::cout << runEn1.size() << " runs with 0 energy in HCAL\n";
  for (std::map<int, counter>::iterator itr = runEn1.begin(); itr != runEn1.end(); ++itr, ++k)
    std::cout << "[" << k << "] Run " << itr->first << " Total " << (itr->second).total << " in p-bins "
              << (itr->second).count[0] << ":" << (itr->second).count[1] << ":" << (itr->second).count[2] << ":"
              << (itr->second).count[3] << std::endl;
  k = 0;
  std::cout << runEn2.size() << " runs with 0 energy in ECAL and HCAL\n";
  for (std::map<int, counter>::iterator itr = runEn2.begin(); itr != runEn2.end(); ++itr, ++k)
    std::cout << "[" << k << "] Run " << itr->first << " Total " << (itr->second).total << " in p-bins "
              << (itr->second).count[0] << ":" << (itr->second).count[1] << ":" << (itr->second).count[2] << ":"
              << (itr->second).count[3] << std::endl;
  if (((flag_ / 100) % 10) > 0) {
    fileout_.close();
    std::cout << "Writes " << good << " events in the file " << outFileName_ << std::endl;
  }
  std::cout << "Finds " << duplicate << " Duplicate events out of " << kount << " evnts in this file with p>20 Gev"
            << std::endl;
  std::cout << "Number of selected events:" << std::endl;
  for (unsigned int k = 1; k < ps_.size(); ++k)
    std::cout << ps_[k - 1] << ":" << ps_[k] << "     " << kounts[k - 1] << std::endl;
  std::cout << "Number in each step: ";
  for (unsigned int k = 0; k < 18; ++k)
    std::cout << " [" << k << "] " << kount50[k];
  std::cout << std::endl;
}

bool CalibMonitor::goodTrack(double &eHcal, double &cuti, const Long64_t &entry, bool debug) {
  bool select(true);
  double cut(cuti);
  if (debug) {
    std::cout << "goodTrack input " << eHcal << ":" << cut;
  }
  if (flexibleSelect_ > 1) {
    double eta = (t_ieta > 0) ? t_ieta : -t_ieta;
    cut = 8.0 * exp(eta * log2by18_);
  }
  correctEnergy(eHcal, entry);
  select = ((t_qltyFlag) && (t_selectTk) && (t_hmaxNearP < cut) && (t_eMipDR < 1.0) && (eHcal > 0.001));
  if (debug) {
    std::cout << " output " << select << " Based on " << t_qltyFlag << ":" << t_selectTk << ":" << t_hmaxNearP << ":"
              << cut << ":" << t_eMipDR << ":" << eHcal << std::endl;
  }
  return select;
}

bool CalibMonitor::selectPhi(bool debug) {
  bool select(true);
  if (phimin_ > 1 || phimax_ < 72) {
    double eTotal(0), eSelec(0);
    // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
    for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
      int iphi = ((*t_DetIds)[k]) & (0x3FF);
      int zside = ((*t_DetIds)[k] & 0x80000) ? (1) : (-1);
      eTotal += ((*t_HitEnergies)[k]);
      if (iphi >= phimin_ && iphi <= phimax_ && zside == zside_)
        eSelec += ((*t_HitEnergies)[k]);
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

void CalibMonitor::plotHist(int itype, int inum, bool save) {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(1);
  char name[100];
  int itmin = (itype >= 0 && itype < 4) ? itype : 0;
  int itmax = (itype >= 0 && itype < 4) ? itype : 3;
  std::string types[4] = {
      "E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})"};
  int nmax[4] = {npbin, npbin, npbin, npbin};
  for (int type = itmin; type <= itmax; ++type) {
    int inmin = (inum >= 0 && inum < nmax[type]) ? inum : 0;
    int inmax = (inum >= 0 && inum < nmax[type]) ? inum : nmax[type] - 1;
    int kmax = 1;
    if (type == 0)
      kmax = (int)(etas_.size());
    else if (type == 1)
      kmax = (int)(etas_.size());
    else if (type == 2)
      kmax = (int)(nvx_.size());
    else
      kmax = (int)(dl1_.size());
    for (int num = inmin; num <= inmax; ++num) {
      for (int k = 0; k < kmax; ++k) {
        sprintf(name, "c_%d%d%d", type, num, k);
        TCanvas *pad = new TCanvas(name, name, 700, 500);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        sprintf(name, "%s", types[type].c_str());
        if (type != 7) {
          TH1D *hist(0);
          if (type == 0)
            hist = (TH1D *)(h_etaR[num][k]->Clone());
          else if (type == 1)
            hist = (TH1D *)(h_etaF[num][k]->Clone());
          else if (type == 2)
            hist = (TH1D *)(h_nvxR[num][k]->Clone());
          else
            hist = (TH1D *)(h_dL1R[num][k]->Clone());
          hist->GetXaxis()->SetTitle(name);
          hist->GetYaxis()->SetTitle("Tracks");
          drawHist(hist, pad);
          if (save) {
            sprintf(name, "c_%s%d%d%d.gif", prefix_.c_str(), type, num, k);
            pad->Print(name);
          }
        } else {
          TProfile *hist = (TProfile *)(h_etaX[num][k]->Clone());
          hist->GetXaxis()->SetTitle(name);
          hist->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
          hist->GetYaxis()->SetRangeUser(0.4, 1.6);
          hist->Fit("pol0", "q");
          drawHist(hist, pad);
          if (save) {
            sprintf(name, "c_%s%d%d%d.gif", prefix_.c_str(), type, num, k);
            pad->Print(name);
          }
        }
      }
    }
  }
}

template <class Hist>
void CalibMonitor::drawHist(Hist *hist, TCanvas *pad) {
  hist->GetYaxis()->SetLabelOffset(0.005);
  hist->GetYaxis()->SetTitleOffset(1.20);
  hist->Draw();
  pad->Update();
  TPaveStats *st1 = (TPaveStats *)hist->GetListOfFunctions()->FindObject("stats");
  if (st1 != NULL) {
    st1->SetY1NDC(0.70);
    st1->SetY2NDC(0.90);
    st1->SetX1NDC(0.55);
    st1->SetX2NDC(0.90);
  }
  pad->Modified();
  pad->Update();
}

void CalibMonitor::savePlot(const std::string &theName, bool append, bool all) {
  TFile *theFile(0);
  if (append) {
    theFile = new TFile(theName.c_str(), "UPDATE");
  } else {
    theFile = new TFile(theName.c_str(), "RECREATE");
  }

  theFile->cd();
  for (unsigned int k = 0; k < ps_.size(); ++k) {
    for (unsigned int j = 0; j <= ietas_.size(); ++j) {
      if ((all || k == kp50) && h_pp[k].size() > j && h_pp[k][j] != 0) {
        TH1D *hist = (TH1D *)h_pp[k][j]->Clone();
        hist->Write();
      }
    }
    if (plotType_ <= 1) {
      for (unsigned int i = 0; i < nvx_.size(); ++i) {
        if (h_etaX[k][i] != 0) {
          TProfile *hnew = (TProfile *)h_etaX[k][i]->Clone();
          hnew->Write();
        }
        if (h_nvxR[k].size() > i && h_nvxR[k][i] != 0) {
          TH1D *hist = (TH1D *)h_nvxR[k][i]->Clone();
          hist->Write();
        }
      }
    }
    for (unsigned int j = 0; j < etas_.size(); ++j) {
      if ((plotType_ <= 1) && (h_etaR[k][j] != 0)) {
        TH1D *hist = (TH1D *)h_etaR[k][j]->Clone();
        hist->Write();
      }
      if (h_etaF[k].size() > j && h_etaF[k][j] != 0 && (all || (k == kp50))) {
        TH1D *hist = (TH1D *)h_etaF[k][j]->Clone();
        hist->Write();
      }
    }
    if (plotType_ <= 1) {
      for (unsigned int j = 0; j < dl1_.size(); ++j) {
        if (h_dL1R[k][j] != 0) {
          TH1D *hist = (TH1D *)h_dL1R[k][j]->Clone();
          hist->Write();
        }
      }
    }
    for (unsigned int j = 0; j < ietas_.size(); ++j) {
      if (h_etaB[k].size() > j && h_etaB[k][j] != 0 && (all || (k == kp50))) {
        TH1D *hist = (TH1D *)h_etaB[k][j]->Clone();
        hist->Write();
      }
    }
  }
  if (selRBX_) {
    for (unsigned int k = 0; k < 18; ++k) {
      if (h_rbx[k] != 0) {
        TH1D *h1 = (TH1D *)h_rbx[k]->Clone();
        h1->Write();
      }
    }
  }
  std::cout << "All done" << std::endl;
  theFile->Close();
}

void CalibMonitor::correctEnergy(double &eHcal, const Long64_t &entry) {
  bool debug(false);
  double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
  if ((ifDepth_ == 3) && (cFactor_ != nullptr)) {
    double cfac = cFactor_->getCorr(entry);
    eHcal *= cfac;
    if (debug)
      std::cout << "PU Factor for " << ifDepth_ << ":" << entry << " = " << cfac << ":" << eHcal << std::endl;
  } else if ((corrPU_ < 0) && (pmom > 0)) {
    double ediff = (t_eHcal30 - t_eHcal10);
    if (t_DetIds1 != 0 && t_DetIds3 != 0) {
      double Etot1(0), Etot3(0);
      // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
      for (unsigned int idet = 0; idet < (*t_DetIds1).size(); idet++) {
        unsigned int id = truncateId((*t_DetIds1)[idet], truncateFlag_, false);
        double cfac = corrFactor_->getCorr(id);
        if (cFactor_ != 0)
          cfac *= cFactor_->getCorr(t_Run, (*t_DetIds1)[idet]);
        double hitEn = cfac * (*t_HitEnergies1)[idet];
        Etot1 += hitEn;
      }
      for (unsigned int idet = 0; idet < (*t_DetIds3).size(); idet++) {
        unsigned int id = truncateId((*t_DetIds3)[idet], truncateFlag_, false);
        double cfac = corrFactor_->getCorr(id);
        if (cFactor_ != 0)
          cfac *= cFactor_->getCorr(t_Run, (*t_DetIds3)[idet]);
        double hitEn = cfac * (*t_HitEnergies3)[idet];
        Etot3 += hitEn;
      }
      ediff = (Etot3 - Etot1);
    }
    double fac = puFactor(-corrPU_, t_ieta, pmom, eHcal, ediff, false);
    if (debug) {
      double fac1 = puFactor(-corrPU_, t_ieta, pmom, eHcal, ediff, true);
      double fac2 = puFactor(2, t_ieta, pmom, eHcal, ediff, true);
      std::cout << "PU Factor for " << -corrPU_ << " = " << fac1 << "; for 2 = " << fac2 << std::endl;
    }
    eHcal *= fac;
  } else if (corrPU_ > 0) {
    eHcal = puFactorRho(corrPU_, t_ieta, t_rhoh, eHcal);
  }
}

class GetEntries {
public:
  TTree *fChain;   //!pointer to the analyzed TTree/TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  // Declaration of leaf types
  UInt_t t_RunNo;
  UInt_t t_EventNo;
  Int_t t_Tracks;
  Int_t t_TracksProp;
  Int_t t_TracksSaved;
  Int_t t_TracksLoose;
  Int_t t_TracksTight;
  Int_t t_allvertex;
  Bool_t t_TrigPass;
  Bool_t t_TrigPassSel;
  Bool_t t_L1Bit;
  std::vector<Bool_t> *t_hltbits;
  std::vector<int> *t_ietaAll;
  std::vector<int> *t_ietaGood;
  std::vector<int> *t_trackType;

  // List of branches
  TBranch *b_t_RunNo;        //!
  TBranch *b_t_EventNo;      //!
  TBranch *b_t_Tracks;       //!
  TBranch *b_t_TracksProp;   //!
  TBranch *b_t_TracksSaved;  //!
  TBranch *b_t_TracksLoose;  //!
  TBranch *b_t_TracksTight;  //!
  TBranch *b_t_allvertex;    //!
  TBranch *b_t_TrigPass;     //!
  TBranch *b_t_TrigPassSel;  //!
  TBranch *b_t_L1Bit;        //!
  TBranch *b_t_hltbits;      //!
  TBranch *b_t_ietaAll;      //!
  TBranch *b_t_ietaGood;     //!
  TBranch *b_t_trackType;    //!

  GetEntries(const std::string &fname,
             const std::string &dirname,
             const char *dupFileName,
             const unsigned int bit1,
             const unsigned int bit2);
  virtual ~GetEntries();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TTree *tree, const char *dupFileName);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);

private:
  unsigned int bit_[2];
  std::vector<Long64_t> entries_;
  TH1I *h_tk[3], *h_eta[4], *h_pvx[3];
  TH1D *h_eff[3];
};

GetEntries::GetEntries(const std::string &fname,
                       const std::string &dirnm,
                       const char *dupFileName,
                       const unsigned int bit1,
                       const unsigned int bit2) {
  TFile *file = new TFile(fname.c_str());
  TDirectory *dir = (TDirectory *)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir << std::endl;
  TTree *tree = (TTree *)dir->Get("EventInfo");
  std::cout << "CalibTree " << tree << std::endl;
  bit_[0] = bit1;
  bit_[1] = bit2;
  Init(tree, dupFileName);
}

GetEntries::~GetEntries() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t GetEntries::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t GetEntries::LoadTree(Long64_t entry) {
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

void GetEntries::Init(TTree *tree, const char *dupFileName) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set branch addresses and branch pointers
  // Set object pointer
  t_hltbits = 0;
  t_ietaAll = 0;
  t_ietaGood = 0;
  t_trackType = 0;
  t_L1Bit = false;
  fChain = tree;
  fCurrent = -1;
  if (!tree)
    return;
  fChain->SetMakeClass(1);
  fChain->SetBranchAddress("t_RunNo", &t_RunNo, &b_t_RunNo);
  fChain->SetBranchAddress("t_EventNo", &t_EventNo, &b_t_EventNo);
  fChain->SetBranchAddress("t_Tracks", &t_Tracks, &b_t_Tracks);
  fChain->SetBranchAddress("t_TracksProp", &t_TracksProp, &b_t_TracksProp);
  fChain->SetBranchAddress("t_TracksSaved", &t_TracksSaved, &b_t_TracksSaved);
  fChain->SetBranchAddress("t_TracksLoose", &t_TracksLoose, &b_t_TracksLoose);
  fChain->SetBranchAddress("t_TracksTight", &t_TracksTight, &b_t_TracksTight);
  fChain->SetBranchAddress("t_allvertex", &t_allvertex, &b_t_allvertex);
  fChain->SetBranchAddress("t_TrigPass", &t_TrigPass, &b_t_TrigPass);
  fChain->SetBranchAddress("t_TrigPassSel", &t_TrigPassSel, &b_t_TrigPassSel);
  fChain->SetBranchAddress("t_L1Bit", &t_L1Bit, &b_t_L1Bit);
  fChain->SetBranchAddress("t_hltbits", &t_hltbits, &b_t_hltbits);
  fChain->SetBranchAddress("t_ietaAll", &t_ietaAll, &b_t_ietaAll);
  fChain->SetBranchAddress("t_ietaGood", &t_ietaGood, &b_t_ietaGood);
  fChain->SetBranchAddress("t_trackType", &t_trackType, &b_t_trackType);
  Notify();

  ifstream infile(dupFileName);
  if (!infile.is_open()) {
    std::cout << "Cannot open " << dupFileName << std::endl;
  } else {
    while (1) {
      Long64_t jentry;
      infile >> jentry;
      if (!infile.good())
        break;
      entries_.push_back(jentry);
    }
    infile.close();
    std::cout << "Reads a list of " << entries_.size() << " events from " << dupFileName << std::endl;
  }

  h_tk[0] = new TH1I("Track0", "# of tracks produced", 2000, 0, 2000);
  h_tk[1] = new TH1I("Track1", "# of tracks propagated", 2000, 0, 2000);
  h_tk[2] = new TH1I("Track2", "# of tracks saved in tree", 2000, 0, 2000);
  h_eta[0] = new TH1I("Eta0", "i#eta (All Tracks)", 60, -30, 30);
  h_eta[1] = new TH1I("Eta1", "i#eta (Good Tracks)", 60, -30, 30);
  h_eta[2] = new TH1I("Eta2", "i#eta (Loose Isolated Tracks)", 60, -30, 30);
  h_eta[3] = new TH1I("Eta3", "i#eta (Tight Isolated Tracks)", 60, -30, 30);
  h_eff[0] = new TH1D("Eff0", "i#eta (Selection Efficiency)", 60, -30, 30);
  h_eff[1] = new TH1D("Eff1", "i#eta (Loose Isolation Efficiency)", 60, -30, 30);
  h_eff[2] = new TH1D("Eff2", "i#eta (Tight Isolation Efficiency)", 60, -30, 30);
  h_pvx[0] = new TH1I("Pvx0", "Number of Good Vertex (All)", 100, 0, 100);
  h_pvx[1] = new TH1I("Pvx1", "Number of Good Vertex (Loose)", 100, 0, 100);
  h_pvx[2] = new TH1I("Pvx2", "Number of Good Vertex (Tight)", 100, 0, 100);
}

Bool_t GetEntries::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void GetEntries::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t GetEntries::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void GetEntries::Loop() {
  //   In a ROOT session, you can do:
  //      Root > .L CalibMonitor.C+g
  //      Root > GetEntries t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
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
  if (fChain == 0)
    return;

  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  unsigned int kount(0), duplicate(0), selected(0);
  int l1(0), hlt(0), loose(0), tight(0);
  int allHLT[3] = {0, 0, 0};
  int looseHLT[3] = {0, 0, 0};
  int tightHLT[3] = {0, 0, 0};
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    bool select = (std::find(entries_.begin(), entries_.end(), jentry) == entries_.end());
    if (!select) {
      ++duplicate;
      continue;
    }
    h_tk[0]->Fill(t_Tracks);
    h_tk[1]->Fill(t_TracksProp);
    h_tk[2]->Fill(t_TracksSaved);
    h_pvx[0]->Fill(t_allvertex);
    if (t_TracksLoose > 0)
      h_pvx[1]->Fill(t_allvertex);
    if (t_TracksTight > 0)
      h_pvx[2]->Fill(t_allvertex);
    if (t_L1Bit) {
      ++l1;
      if (t_TracksLoose > 0)
        ++loose;
      if (t_TracksTight > 0)
        ++tight;
      if (t_TrigPass)
        ++hlt;
    }
    if (t_TrigPass) {
      ++kount;
      if (t_TrigPassSel)
        ++selected;
    }
    bool passt[2] = {false, false}, pass(false);
    for (unsigned k = 0; k < t_hltbits->size(); ++k) {
      if ((*t_hltbits)[k] > 0) {
        pass = true;
        for (int i = 0; i < 2; ++i)
          if (k == bit_[i])
            passt[i] = true;
      }
    }
    if (pass) {
      ++allHLT[0];
      for (int i = 0; i < 2; ++i)
        if (passt[i])
          ++allHLT[i + 1];
      if (t_TracksLoose > 0) {
        ++looseHLT[0];
        for (int i = 0; i < 2; ++i)
          if (passt[i])
            ++looseHLT[i + 1];
      }
      if (t_TracksTight > 0) {
        ++tightHLT[0];
        for (int i = 0; i < 2; ++i)
          if (passt[i])
            ++tightHLT[i + 1];
      }
    }
    for (unsigned int k = 0; k < t_ietaAll->size(); ++k)
      h_eta[0]->Fill((*t_ietaAll)[k]);
    for (unsigned int k = 0; k < t_ietaGood->size(); ++k) {
      h_eta[1]->Fill((*t_ietaGood)[k]);
      if (t_trackType->size() > 0) {
        if ((*t_trackType)[k] > 0)
          h_eta[2]->Fill((*t_ietaGood)[k]);
        if ((*t_trackType)[k] > 1)
          h_eta[3]->Fill((*t_ietaGood)[k]);
      }
    }
  }
  double ymaxk(0);
  for (int i = 1; i <= h_eff[0]->GetNbinsX(); ++i) {
    double rat(0), drat(0);
    if (h_eta[0]->GetBinContent(i) > ymaxk)
      ymaxk = h_eta[0]->GetBinContent(i);
    if ((h_eta[1]->GetBinContent(i) > 0) && (h_eta[0]->GetBinContent(i) > 0)) {
      rat = h_eta[1]->GetBinContent(i) / h_eta[0]->GetBinContent(i);
      drat = rat * std::sqrt(pow((h_eta[1]->GetBinError(i) / h_eta[1]->GetBinContent(i)), 2) +
                             pow((h_eta[0]->GetBinError(i) / h_eta[0]->GetBinContent(i)), 2));
    }
    h_eff[0]->SetBinContent(i, rat);
    h_eff[0]->SetBinError(i, drat);
    if ((h_eta[1]->GetBinContent(i) > 0) && (h_eta[2]->GetBinContent(i) > 0)) {
      rat = h_eta[2]->GetBinContent(i) / h_eta[1]->GetBinContent(i);
      drat = rat * std::sqrt(pow((h_eta[2]->GetBinError(i) / h_eta[2]->GetBinContent(i)), 2) +
                             pow((h_eta[1]->GetBinError(i) / h_eta[1]->GetBinContent(i)), 2));
    } else {
      rat = drat = 0;
    }
    h_eff[1]->SetBinContent(i, rat);
    h_eff[1]->SetBinError(i, drat);
    if ((h_eta[1]->GetBinContent(i) > 0) && (h_eta[3]->GetBinContent(i) > 0)) {
      rat = h_eta[3]->GetBinContent(i) / h_eta[1]->GetBinContent(i);
      drat = rat * std::sqrt(pow((h_eta[3]->GetBinError(i) / h_eta[3]->GetBinContent(i)), 2) +
                             pow((h_eta[1]->GetBinError(i) / h_eta[1]->GetBinContent(i)), 2));
    } else {
      rat = drat = 0;
    }
    h_eff[1]->SetBinContent(i, rat);
    h_eff[1]->SetBinError(i, drat);
  }
  std::cout << "===== Remove " << duplicate << " events from " << nentries << "\n===== " << kount
            << " events passed trigger of which " << selected << " events get selected =====\n"
            << std::endl;
  std::cout << "===== " << l1 << " events passed L1 " << hlt << " events passed HLT and " << loose << ":" << tight
            << " events have at least 1 track candidate with loose:tight"
            << " isolation cut =====\n"
            << std::endl;
  for (int i = 0; i < 3; ++i) {
    char tbit[20];
    if (i == 0)
      sprintf(tbit, "Any");
    else
      sprintf(tbit, "%3d", bit_[i - 1]);
    std::cout << "Satisfying HLT trigger bit " << tbit << " Kount " << allHLT[i] << " Loose " << looseHLT[i]
              << " Tight " << tightHLT[i] << std::endl;
  }

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(1110);
  gStyle->SetOptTitle(0);
  int color[5] = {kBlack, kRed, kBlue, kMagenta, kCyan};
  int lines[5] = {1, 2, 3, 4, 5};
  TCanvas *pad1 = new TCanvas("c_track", "c_track", 500, 500);
  pad1->SetRightMargin(0.10);
  pad1->SetTopMargin(0.10);
  pad1->SetFillColor(kWhite);
  std::string titl1[3] = {"Reconstructed", "Propagated", "Saved"};
  TLegend *legend1 = new TLegend(0.11, 0.80, 0.50, 0.89);
  legend1->SetFillColor(kWhite);
  double ymax(0), xmax(0);
  for (int k = 0; k < 3; ++k) {
    int total(0), totaltk(0);
    for (int i = 1; i <= h_tk[k]->GetNbinsX(); ++i) {
      if (ymax < h_tk[k]->GetBinContent(i))
        ymax = h_tk[k]->GetBinContent(i);
      if (i > 1)
        total += (int)(h_tk[k]->GetBinContent(i));
      totaltk += (int)(h_tk[k]->GetBinContent(i)) * (i - 1);
      if (h_tk[k]->GetBinContent(i) > 0) {
        if (xmax < h_tk[k]->GetBinLowEdge(i) + h_tk[k]->GetBinWidth(i))
          xmax = h_tk[k]->GetBinLowEdge(i) + h_tk[k]->GetBinWidth(i);
      }
    }
    h_tk[k]->SetLineColor(color[k]);
    h_tk[k]->SetMarkerColor(color[k]);
    h_tk[k]->SetLineStyle(lines[k]);
    std::cout << h_tk[k]->GetTitle() << " Entries " << h_tk[k]->GetEntries() << " Events " << total << " Tracks "
              << totaltk << std::endl;
    legend1->AddEntry(h_tk[k], titl1[k].c_str(), "l");
  }
  int i1 = (int)(0.1 * xmax) + 1;
  xmax = 10.0 * i1;
  int i2 = (int)(0.01 * ymax) + 1;

  ymax = 100.0 * i2;
  for (int k = 0; k < 3; ++k) {
    h_tk[k]->GetXaxis()->SetRangeUser(0, xmax);
    h_tk[k]->GetYaxis()->SetRangeUser(0.1, ymax);
    h_tk[k]->GetXaxis()->SetTitle("# Tracks");
    h_tk[k]->GetYaxis()->SetTitle("Events");
    h_tk[k]->GetYaxis()->SetLabelOffset(0.005);
    h_tk[k]->GetYaxis()->SetTitleOffset(1.20);
    if (k == 0)
      h_tk[k]->Draw("hist");
    else
      h_tk[k]->Draw("hist sames");
  }
  pad1->Update();
  pad1->SetLogy();
  ymax = 0.90;
  for (int k = 0; k < 3; ++k) {
    TPaveStats *st1 = (TPaveStats *)h_tk[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      st1->SetLineColor(color[k]);
      st1->SetTextColor(color[k]);
      st1->SetY1NDC(ymax - 0.09);
      st1->SetY2NDC(ymax);
      st1->SetX1NDC(0.55);
      st1->SetX2NDC(0.90);
      ymax -= 0.09;
    }
  }
  pad1->Modified();
  pad1->Update();
  legend1->Draw("same");
  pad1->Update();

  TCanvas *pad2 = new TCanvas("c_ieta", "c_ieta", 500, 500);
  pad2->SetRightMargin(0.10);
  pad2->SetTopMargin(0.10);
  pad2->SetFillColor(kWhite);
  pad2->SetLogy();
  std::string titl2[4] = {"All Tracks", "Selected Tracks", "Loose Isolation", "Tight Isolation"};
  TLegend *legend2 = new TLegend(0.11, 0.75, 0.50, 0.89);
  legend2->SetFillColor(kWhite);
  i2 = (int)(0.001 * ymaxk) + 1;
  ymax = 1000.0 * i2;
  for (int k = 0; k < 4; ++k) {
    h_eta[k]->GetYaxis()->SetRangeUser(1, ymax);
    h_eta[k]->SetLineColor(color[k]);
    h_eta[k]->SetMarkerColor(color[k]);
    h_eta[k]->SetLineStyle(lines[k]);
    h_eta[k]->GetXaxis()->SetTitle("i#eta");
    h_eta[k]->GetYaxis()->SetTitle("Tracks");
    h_eta[k]->GetYaxis()->SetLabelOffset(0.005);
    h_eta[k]->GetYaxis()->SetTitleOffset(1.20);
    legend2->AddEntry(h_eta[k], titl2[k].c_str(), "l");
    if (k == 0)
      h_eta[k]->Draw("hist");
    else
      h_eta[k]->Draw("hist sames");
  }
  pad2->Update();
  ymax = 0.90;
  //double ymin = 0.10;
  for (int k = 0; k < 4; ++k) {
    TPaveStats *st1 = (TPaveStats *)h_eta[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      st1->SetLineColor(color[k]);
      st1->SetTextColor(color[k]);
      st1->SetY1NDC(ymax - 0.09);
      st1->SetY2NDC(ymax);
      st1->SetX1NDC(0.55);
      st1->SetX2NDC(0.90);
      ymax -= 0.09;
    }
  }
  pad2->Modified();
  pad2->Update();
  legend2->Draw("same");
  pad2->Update();

  std::string titl3[3] = {"Selection", "Loose Isolation", "Tight Isolation"};
  TCanvas *pad3 = new TCanvas("c_effi", "c_effi", 500, 500);
  TLegend *legend3 = new TLegend(0.11, 0.785, 0.50, 0.89);
  pad3->SetRightMargin(0.10);
  pad3->SetTopMargin(0.10);
  pad3->SetFillColor(kWhite);
  pad3->SetLogy();
  for (int k = 0; k < 3; ++k) {
    h_eff[k]->SetStats(0);
    h_eff[k]->SetMarkerStyle(20);
    h_eff[k]->SetLineColor(color[k]);
    h_eff[k]->SetMarkerColor(color[k]);
    h_eff[k]->GetXaxis()->SetTitle("i#eta");
    h_eff[k]->GetYaxis()->SetTitle("Efficiency");
    h_eff[k]->GetYaxis()->SetLabelOffset(0.005);
    h_eff[k]->GetYaxis()->SetTitleOffset(1.20);
    if (k == 0)
      h_eff[k]->Draw("");
    else
      h_eff[k]->Draw("same");
    legend3->AddEntry(h_eff[k], titl3[k].c_str(), "l");
  }
  pad3->Modified();
  pad3->Update();
  legend3->Draw("same");
  pad3->Update();

  std::string titl4[3] = {"All events", "Events with Loose Isolation", "Events with Tight Isolation"};
  TLegend *legend4 = new TLegend(0.11, 0.785, 0.50, 0.89);
  TCanvas *pad4 = new TCanvas("c_nvx", "c_nvx", 500, 500);
  pad4->SetRightMargin(0.10);
  pad4->SetTopMargin(0.10);
  pad4->SetFillColor(kWhite);
  pad4->SetLogy();
  for (int k = 0; k < 3; ++k) {
    h_pvx[k]->SetStats(1110);
    h_pvx[k]->SetMarkerStyle(20);
    h_pvx[k]->SetLineColor(color[k]);
    h_pvx[k]->SetMarkerColor(color[k]);
    h_pvx[k]->GetXaxis()->SetTitle("N_{PV}");
    h_pvx[k]->GetYaxis()->SetTitle("Events");
    h_pvx[k]->GetYaxis()->SetLabelOffset(0.005);
    h_pvx[k]->GetYaxis()->SetTitleOffset(1.20);
    if (k == 0)
      h_pvx[k]->Draw("");
    else
      h_pvx[k]->Draw("sames");
    legend4->AddEntry(h_pvx[k], titl4[k].c_str(), "l");
  }
  pad4->Update();
  ymax = 0.90;
  for (int k = 0; k < 3; ++k) {
    TPaveStats *st1 = (TPaveStats *)h_pvx[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      st1->SetLineColor(color[k]);
      st1->SetTextColor(color[k]);
      st1->SetY1NDC(ymax - 0.09);
      st1->SetY2NDC(ymax);
      st1->SetX1NDC(0.55);
      st1->SetX2NDC(0.90);
      ymax -= 0.09;
    }
  }
  pad4->Modified();
  pad4->Update();
  legend4->Draw("same");
  pad4->Update();
}

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
                      int puCorr = -2,
                      int flag = 1031,
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
                      int rbx = 0,
                      bool exclude = false,
                      bool etamax = false);
  virtual ~CalibPlotProperties();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *, const char *);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  bool goodTrack(double &eHcal, double &cut, const Long64_t &entry, bool debug);
  bool selectPhi(bool debug);
  void savePlot(const std::string &theName, bool append = true, bool all = false);
  void correctEnergy(double &ener, const Long64_t &entry);

private:
  static const unsigned int npbin = 6, kp50 = 3, ndepth = 7;
  CalibCorrFactor *corrFactor_;
  CalibCorr *cFactor_;
  CalibSelectRBX *cSelect_;
  const std::string fname_, dirnm_, prefix_, outFileName_;
  const int corrPU_, flag_;
  const bool dataMC_, useGen_;
  const int truncateFlag_;
  const int etalo_, etahi_;
  int runlo_, runhi_;
  const int phimin_, phimax_, zside_, nvxlo_, nvxhi_, rbx_;
  int ifDepth_;
  bool exclude_, corrE_, cutL1T_;
  bool includeRun_, getHist_;
  int flexibleSelect_;
  bool plotBasic_, plotEnergy_, plotHists_;
  double log2by18_;
  std::ofstream fileout_;
  std::vector<Long64_t> entries_;
  std::vector<std::pair<int, int> > events_;
  std::vector<double> etas_, ps_, dl1_;
  std::vector<int> nvx_, ietas_;
  TH1D *h_p[5], *h_eta[5], *h_nvtx;
  std::vector<TH1D *> h_eta0, h_eta1, h_eta2, h_eta3, h_eta4;
  std::vector<TH1D *> h_dL1, h_vtx;
  std::vector<TH1D *> h_etaEH[npbin], h_etaEp[npbin], h_etaEE[npbin];
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
                                         int rbx,
                                         bool exc,
                                         bool etam)
    : corrFactor_(nullptr),
      cFactor_(nullptr),
      cSelect_(nullptr),
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
      rbx_(rbx),
      exclude_(exc),
      includeRun_(true) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree

  flexibleSelect_ = ((flag_ / 1) % 10);
  plotBasic_ = (((flag_ / 10) % 10) > 0);
  plotEnergy_ = (((flag_ / 10) % 10) > 0);
  cutL1T_ = ((flag_ / 1000) % 10);
  ifDepth_ = ((flag_ / 10000) % 10);
  plotHists_ = (((flag_ / 100000) % 10) > 0);
  log2by18_ = std::log(2.5) / 18.0;
  if (runlo_ < 0 || runhi_ < 0) {
    runlo_ = std::abs(runlo_);
    runhi_ = std::abs(runhi_);
    includeRun_ = false;
  }
  char treeName[400];
  sprintf(treeName, "%s/CalibTree", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << " flags " << flexibleSelect_ << "|"
            << plotBasic_ << "|"
            << "|" << plotEnergy_ << "|" << plotHists_ << "|" << corrPU_ << "\n cons " << log2by18_ << " eta range "
            << etalo_ << ":" << etahi_ << " run range " << runlo_ << ":" << runhi_ << " (inclusion flag " << includeRun_
            << ")\n Selection of RBX" << rbx << " Vertex Range " << nvxlo_ << ":" << nvxhi_
            << "\n corrFileName: " << corrFileName << " useScale " << useScale << ":" << scl << ":" << etam
            << "\n rcorFileName: " << rcorFileName << " flag " << ifDepth_ << std::endl;
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain, dupFileName);
    corrFactor_ = new CalibCorrFactor(corrFileName, useScale, scl, etam, false);
    if (std::string(rcorFileName) != "") {
      cFactor_ = new CalibCorr(rcorFileName, ifDepth_, false);
    } else {
      ifDepth_ = 0;
    }
    if (rbx != 0)
      cSelect_ = new CalibSelectRBX(rbx, false);
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

void CalibPlotProperties::Init(TChain *tree, const char *dupFileName) {
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

  if (std::string(dupFileName) != "") {
    ifstream infil1(dupFileName);
    if (!infil1.is_open()) {
      std::cout << "Cannot open duplicate file " << dupFileName << std::endl;
    } else {
      while (1) {
        Long64_t jentry;
        infil1 >> jentry;
        if (!infil1.good())
          break;
        entries_.push_back(jentry);
      }
      infil1.close();
      std::cout << "Reads a list of " << entries_.size() << " events from " << dupFileName << std::endl;
    }
  } else {
    std::cout << "No duplicate events in the input file" << std::endl;
  }

  int ipbin[npbin] = {10, 20, 30, 40, 60, 100};
  for (unsigned int i = 0; i < npbin; ++i)
    ps_.push_back((double)(ipbin[i]));
  int ietas[4] = {0, 13, 18, 23};
  for (int i = 0; i < 4; ++i)
    ietas_.push_back(ietas[i]);

  char name[20], title[200];
  unsigned int kp = ps_.size() - 1;
  unsigned int kk(0);
  std::string titl[5] = {
      "all tracks", "good quality tracks", "selected tracks", "isolated good tracks", "tracks having MIP in ECAL"};

  if (plotBasic_) {
    std::cout << "Book Basic Histos" << std::endl;
    for (int k = 0; k < 5; ++k) {
      sprintf(name, "%sp%d", prefix_.c_str(), k);
      sprintf(title, "Momentum for %s", titl[k].c_str());
      h_p[k] = new TH1D(name, title, 100, 10.0, 110.0);
      sprintf(name, "%seta%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s", titl[k].c_str());
      h_eta[k] = new TH1D(name, title, 60, -30.0, 30.0);
    }
    for (unsigned int k = 0; k < kp; ++k) {
      sprintf(name, "%seta0%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s (p = %d:%d GeV)", titl[0].c_str(), ipbin[k], ipbin[k + 1]);
      h_eta0.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta0.size() - 1;
      h_eta0[kk]->Sumw2();
      sprintf(name, "%seta1%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s (p = %d:%d GeV)", titl[1].c_str(), ipbin[k], ipbin[k + 1]);
      h_eta1.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta1.size() - 1;
      h_eta1[kk]->Sumw2();
      sprintf(name, "%seta2%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s (p = %d:%d GeV)", titl[2].c_str(), ipbin[k], ipbin[k + 1]);
      h_eta2.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta2.size() - 1;
      h_eta2[kk]->Sumw2();
      sprintf(name, "%seta3%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s (p = %d:%d GeV)", titl[3].c_str(), ipbin[k], ipbin[k + 1]);
      h_eta3.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta3.size() - 1;
      h_eta3[kk]->Sumw2();
      sprintf(name, "%seta4%d", prefix_.c_str(), k);
      sprintf(title, "#eta for %s (p = %d:%d GeV)", titl[4].c_str(), ipbin[k], ipbin[k + 1]);
      h_eta4.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      kk = h_eta4.size() - 1;
      h_eta4[kk]->Sumw2();
      sprintf(name, "%sdl1%d", prefix_.c_str(), k);
      sprintf(title, "Distance from L1 (p = %d:%d GeV)", ipbin[k], ipbin[k + 1]);
      h_dL1.push_back(new TH1D(name, title, 160, 0.0, 8.0));
      kk = h_dL1.size() - 1;
      h_dL1[kk]->Sumw2();
      sprintf(name, "%svtx%d", prefix_.c_str(), k);
      sprintf(title, "N_{Vertex} (p = %d:%d GeV)", ipbin[k], ipbin[k + 1]);
      h_vtx.push_back(new TH1D(name, title, 100, 0.0, 100.0));
      kk = h_vtx.size() - 1;
      h_vtx[kk]->Sumw2();
    }
  }

  if (plotEnergy_) {
    std::cout << "Make plots for good tracks" << std::endl;
    for (unsigned int k = 0; k < kp; ++k) {
      for (int j = etalo_; j <= etahi_ + 1; ++j) {
        sprintf(name, "%senergyH%d%d", prefix_.c_str(), k, j);
        if (j > etahi_)
          sprintf(title,
                  "HCAL energy for %s (p = %d:%d GeV |#eta| = %d:%d)",
                  titl[3].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  etalo_,
                  etahi_);
        else
          sprintf(title, "HCAL energy for %s (p = %d:%d GeV |#eta| = %d)", titl[3].c_str(), ipbin[k], ipbin[k + 1], j);
        h_etaEH[k].push_back(new TH1D(name, title, 200, 0, 200));
        kk = h_etaEH[k].size() - 1;
        h_etaEH[k][kk]->Sumw2();
        sprintf(name, "%senergyP%d%d", prefix_.c_str(), k, j);
        if (j > etahi_)
          sprintf(title,
                  "momentum for %s (p = %d:%d GeV |#eta| = %d:%d)",
                  titl[3].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  etalo_,
                  etahi_);
        else
          sprintf(title, "momentum for %s (p = %d:%d GeV |#eta| = %d)", titl[3].c_str(), ipbin[k], ipbin[k + 1], j);
        h_etaEp[k].push_back(new TH1D(name, title, 100, 0, 100));
        kk = h_etaEp[k].size() - 1;
        h_etaEp[k][kk]->Sumw2();
        sprintf(name, "%senergyE%d%d", prefix_.c_str(), k, j);
        if (j > etahi_)
          sprintf(title,
                  "ECAL energy for %s (p = %d:%d GeV |#eta| = %d:%d)",
                  titl[3].c_str(),
                  ipbin[k],
                  ipbin[k + 1],
                  etalo_,
                  etahi_);
        else
          sprintf(title, "ECAL energy for %s (p = %d:%d GeV |#eta| = %d)", titl[3].c_str(), ipbin[k], ipbin[k + 1], j);
        h_etaEE[k].push_back(new TH1D(name, title, 100, 0, 10));
        kk = h_etaEE[k].size() - 1;
        h_etaEE[k][kk]->Sumw2();
      }
    }

    for (unsigned int j = 0; j < ietas_.size(); ++j) {
      sprintf(name, "%senergyH%d", prefix_.c_str(), j);
      if (j == 0)
        sprintf(title, "HCAL energy for %s (All)", titl[3].c_str());
      else
        sprintf(title, "HCAL energy for %s (|#eta| = %d:%d)", titl[3].c_str(), ietas_[j - 1], ietas_[j]);
      h_eHcal.push_back(new TH1D(name, title, 200, 0, 200));
      kk = h_eHcal.size() - 1;
      h_eHcal[kk]->Sumw2();
      sprintf(name, "%senergyP%d", prefix_.c_str(), j);
      if (j == 0)
        sprintf(title, "Track momentum for %s (All)", titl[3].c_str());
      else
        sprintf(title, "Track momentum for %s (|#eta| = %d:%d)", titl[3].c_str(), ietas_[j - 1], ietas_[j]);
      h_mom.push_back(new TH1D(name, title, 100, 0, 100));
      kk = h_mom.size() - 1;
      h_mom[kk]->Sumw2();
      sprintf(name, "%senergyE%d", prefix_.c_str(), j);
      if (j == 0)
        sprintf(title, "ECAL energy for %s (All)", titl[3].c_str());
      else
        sprintf(title, "ECAL energy for %s (|#eta| = %d:%d)", titl[3].c_str(), ietas_[j - 1], ietas_[j]);
      h_eEcal.push_back(new TH1D(name, title, 100, 0, 10));
      kk = h_eEcal.size() - 1;
      h_eEcal[kk]->Sumw2();
    }
  }

  if (plotHists_) {
    h_nvtx = new TH1D("hnvtx", "Number of vertices", 10, 0, 100);
    h_nvtx->Sumw2();
    for (unsigned int i = 0; i < ndepth; i++) {
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
    h_etaE = new TH2F("heta", "", 50, -25, 25, 100, 0, 100);
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

void CalibPlotProperties::Loop() {
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
  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << std::endl;
  Long64_t nbytes(0), nb(0);
  unsigned int duplicate(0), good(0), kount(0);
  double sel(0), selHB(0), selHE(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (jentry % 100000 == 0)
      std::cout << "Entry " << jentry << " Run " << t_Run << " Event " << t_Event << std::endl;
    bool select = (std::find(entries_.begin(), entries_.end(), jentry) == entries_.end());
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

    if ((ifDepth_ == 3) && (cFactor_ != nullptr) && (cFactor_->absent(ientry)))
      continue;
    // if (Cut(ientry) < 0) continue;
    unsigned int kp = ps_.size();
    double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
    for (unsigned int k = 1; k < ps_.size(); ++k) {
      if (pmom >= ps_[k - 1] && pmom < ps_[k]) {
        kp = k - 1;
        break;
      }
    }
    int jp1 = (((std::abs(t_ieta) >= etalo_) && (std::abs(t_ieta) <= etahi_)) ? (std::abs(t_ieta) - etalo_) : -1);
    int jp2 = (etahi_ - etalo_ + 1);
    unsigned int je1 = ietas_.size();
    for (unsigned int j = 1; j < ietas_.size(); ++j) {
      if (std::abs(t_ieta) > ietas_[j - 1] && std::abs(t_ieta) <= ietas_[j]) {
        je1 = j;
        break;
      }
    }
    int je2 = (je1 != ietas_.size()) ? 0 : -1;
    if (debug)
      std::cout << "Bin " << kp << ":" << je1 << ":" << je2 << ":" << jp1 << ":" << jp2 << std::endl;
    double cut = (pmom > 20) ? ((flexibleSelect_ == 0) ? 2.0 : 10.0) : 0.0;
    double rcut(-1000.0);

    // Selection of good track and energy measured in Hcal
    double eHcal(t_eHcal);
    if (corrFactor_->doCorr() || (cFactor_ != 0)) {
      eHcal = 0;
      for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
        // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
        double cfac(1.0);
        if (corrFactor_->doCorr()) {
          unsigned int id = truncateId((*t_DetIds)[k], truncateFlag_, false);
          cfac = corrFactor_->getCorr(id);
        }
        if (cFactor_ != 0)
          cfac *= cFactor_->getCorr(t_Run, (*t_DetIds)[k]);
        eHcal += (cfac * ((*t_HitEnergies)[k]));
        if (debug) {
          int subdet, zside, ieta, iphi, depth;
          unpackDetId((*t_DetIds)[k], subdet, zside, ieta, iphi, depth);
          std::cout << zside << ":" << ieta << ":" << depth << " Corr " << cfac << " " << (*t_HitEnergies)[k] << " Out "
                    << eHcal << std::endl;
        }
      }
    }
    bool goodTk = goodTrack(eHcal, cut, jentry, debug);
    bool selPhi = selectPhi(debug);
    double rat = (pmom > 0) ? (eHcal / (pmom - t_eMipDR)) : 1.0;
    if (debug)
      std::cout << "Entry " << jentry << " p|eHcal|ratio " << pmom << "|" << t_eHcal << "|" << eHcal << "|" << rat
                << "|" << kp << " Cuts " << t_qltyFlag << "|" << t_selectTk << "|" << (t_hmaxNearP < cut) << "|"
                << (t_eMipDR < 1.0) << "|" << goodTk << "|" << (rat > rcut) << " Select Phi " << selPhi << std::endl;
    if (plotBasic_) {
      h_p[0]->Fill(pmom, t_EventWeight);
      h_eta[0]->Fill(t_ieta, t_EventWeight);
      if (kp < ps_.size())
        h_eta0[kp]->Fill(t_ieta, t_EventWeight);
      if (t_qltyFlag) {
        h_p[1]->Fill(pmom, t_EventWeight);
        h_eta[1]->Fill(t_ieta, t_EventWeight);
        if (kp < ps_.size())
          h_eta1[kp]->Fill(t_ieta, t_EventWeight);
        if (t_selectTk) {
          h_p[2]->Fill(pmom, t_EventWeight);
          h_eta[2]->Fill(t_ieta, t_EventWeight);
          if (kp < ps_.size())
            h_eta2[kp]->Fill(t_ieta, t_EventWeight);
          if (t_hmaxNearP < cut) {
            h_p[3]->Fill(pmom, t_EventWeight);
            h_eta[3]->Fill(t_ieta, t_EventWeight);
            if (kp < ps_.size())
              h_eta3[kp]->Fill(t_ieta, t_EventWeight);
            if (t_eMipDR < 1.0) {
              h_p[4]->Fill(pmom, t_EventWeight);
              h_eta[4]->Fill(t_ieta, t_EventWeight);
              if (kp < ps_.size()) {
                h_eta4[kp]->Fill(t_ieta, t_EventWeight);
                h_dL1[kp]->Fill(t_mindR1, t_EventWeight);
                h_vtx[kp]->Fill(t_goodPV, t_EventWeight);
              }
            }
          }
        }
      }
    }

    if (goodTk && kp < ps_.size() && selPhi) {
      if (rat > rcut) {
        if (plotEnergy_) {
          if (jp1 >= 0) {
            h_etaEH[kp][jp1]->Fill(eHcal, t_EventWeight);
            h_etaEH[kp][jp2]->Fill(eHcal, t_EventWeight);
            h_etaEp[kp][jp1]->Fill(pmom, t_EventWeight);
            h_etaEp[kp][jp2]->Fill(pmom, t_EventWeight);
            h_etaEE[kp][jp1]->Fill(t_eMipDR, t_EventWeight);
            h_etaEE[kp][jp2]->Fill(t_eMipDR, t_EventWeight);
          }
          if (kp == kp50) {
            if (je1 != ietas_.size()) {
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
          h_nvtx->Fill(t_nVtx);
          bool bad(false);
          for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
            unsigned int id = truncateId((*t_DetIds)[k], truncateFlag_, false);
            double cfac = corrFactor_->getCorr(id);
            if (cFactor_ != 0)
              cfac *= cFactor_->getCorr(t_Run, (*t_DetIds)[k]);
            double ener = cfac * (*t_HitEnergies)[k];
            if (corrPU_)
              correctEnergy(ener, jentry);
            if (ener < 0.001)
              bad = true;
          }
          if ((!bad) && (std::fabs(rat - 1) < 0.15) && (kp == kp50) &&
              ((std::abs(t_ieta) < 15) || (std::abs(t_ieta) > 17))) {
            float weight = (dataMC_ ? t_EventWeight : t_EventWeight * puweight(t_nVtx));
            h_etaE->Fill(t_ieta, eHcal, weight);
            sel += weight;
            std::vector<float> bv(7, 0.0f), ev(7, 0.0f);
            std::vector<int> bnrec(7, 0), enrec(7, 0);
            double eb(0), ee(0);
            for (unsigned int k = 0; k < t_HitEnergies->size(); ++k) {
              unsigned int id = truncateId((*t_DetIds)[k], truncateFlag_, false);
              double cfac = corrFactor_->getCorr(id);
              if (cFactor_ != 0)
                cfac *= cFactor_->getCorr(t_Run, (*t_DetIds)[k]);
              double ener = cfac * (*t_HitEnergies)[k];
              if (corrPU_)
                correctEnergy(ener, jentry);
              unsigned int idx = (unsigned int)((*t_DetIds)[k]);
              int subdet, zside, ieta, iphi, depth;
              unpackDetId(idx, subdet, zside, ieta, iphi, depth);
              if (depth > 0 && depth <= (int)(ndepth)) {
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
            bool barrel = (eb > ee);
            if (barrel)
              selHB += weight;
            else
              selHE += weight;
            for (unsigned int i = 0; i < ndepth; i++) {
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

bool CalibPlotProperties::goodTrack(double &eHcal, double &cuti, const Long64_t &entry, bool debug) {
  bool select(true);
  double cut(cuti);
  if (debug) {
    std::cout << "goodTrack input " << eHcal << ":" << cut;
  }
  if (flexibleSelect_ > 1) {
    double eta = (t_ieta > 0) ? t_ieta : -t_ieta;
    cut = 8.0 * exp(eta * log2by18_);
  }
  correctEnergy(eHcal, entry);
  select = ((t_qltyFlag) && (t_selectTk) && (t_hmaxNearP < cut) && (t_eMipDR < 100.0) && (eHcal > 0.001));
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
      int iphi = ((*t_DetIds)[k]) & (0x3FF);
      int zside = ((*t_DetIds)[k] & 0x80000) ? (1) : (-1);
      eTotal += ((*t_HitEnergies)[k]);
      if (iphi >= phimin_ && iphi <= phimax_ && zside == zside_)
        eSelec += ((*t_HitEnergies)[k]);
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

void CalibPlotProperties::savePlot(const std::string &theName, bool append, bool all) {
  TFile *theFile(0);
  if (append) {
    theFile = new TFile(theName.c_str(), "UPDATE");
  } else {
    theFile = new TFile(theName.c_str(), "RECREATE");
  }

  theFile->cd();

  if (plotBasic_) {
    for (int k = 0; k < 5; ++k) {
      if (h_p[k] != 0) {
        TH1D *h1 = (TH1D *)h_p[k]->Clone();
        h1->Write();
      }
      if (h_eta[k] != 0) {
        TH1D *h2 = (TH1D *)h_eta[k]->Clone();
        h2->Write();
      }
    }
    for (unsigned int k = 0; k < h_eta0.size(); ++k) {
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
    for (unsigned int k = 0; k < ps_.size() - 1; ++k) {
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
      }
    }

    for (unsigned int j = 0; j < ietas_.size(); ++j) {
      if (h_eHcal.size() > j && (h_eHcal[j] != nullptr)) {
        TH1D *hist = (TH1D *)h_eHcal[j]->Clone();
        hist->Write();
      }
      if (h_mom.size() > j && (h_mom[j] != nullptr)) {
        TH1D *hist = (TH1D *)h_mom[j]->Clone();
        hist->Write();
      }
      if (h_eEcal.size() > j && (h_eEcal[j] != nullptr)) {
        TH1D *hist = (TH1D *)h_eEcal[j]->Clone();
        hist->Write();
      }
    }
  }

  if (plotHists_) {
    if (h_nvtx != 0) {
      TH1D *h1 = (TH1D *)h_nvtx->Clone();
      h1->Write();
    }
    if (h_etaE != 0) {
      TH2D *h2 = (TH2D *)h_etaE->Clone();
      h2->Write();
    }
    for (unsigned int i = 0; i < ndepth; ++i) {
      if (h_bvlist[i] != 0) {
        TH1D *h1 = (TH1D *)h_bvlist[i]->Clone();
        h1->Write();
      }
      if (h_bvlist2[i] != 0) {
        TH1D *h2 = (TH1D *)h_bvlist2[i]->Clone();
        h2->Write();
      }
      if (h_bvlist3[i] != 0) {
        TH1D *h3 = (TH1D *)h_bvlist3[i]->Clone();
        h3->Write();
      }
      if (h_evlist[i] != 0) {
        TH1D *h4 = (TH1D *)h_evlist[i]->Clone();
        h4->Write();
      }
      if (h_evlist2[i] != 0) {
        TH1D *h5 = (TH1D *)h_evlist2[i]->Clone();
        h5->Write();
      }
      if (h_evlist3[i] != 0) {
        TH1D *h6 = (TH1D *)h_evlist3[i]->Clone();
        h6->Write();
      }
    }
  }
  std::cout << "All done" << std::endl;
  theFile->Close();
}

void CalibPlotProperties::correctEnergy(double &eHcal, const Long64_t &entry) {
  double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
  if ((ifDepth_ == 3) && (cFactor_ != nullptr)) {
    double cfac = cFactor_->getCorr(entry);
    eHcal *= cfac;
  } else if ((corrPU_ < 0) && (pmom > 0)) {
    double ediff = (t_eHcal30 - t_eHcal10);
    if (t_DetIds1 != 0 && t_DetIds3 != 0) {
      double Etot1(0), Etot3(0);
      // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
      for (unsigned int idet = 0; idet < (*t_DetIds1).size(); idet++) {
        unsigned int id = truncateId((*t_DetIds1)[idet], truncateFlag_, false);
        double cfac = corrFactor_->getCorr(id);
        if (cFactor_ != 0)
          cfac *= cFactor_->getCorr(t_Run, (*t_DetIds1)[idet]);
        double hitEn = cfac * (*t_HitEnergies1)[idet];
        Etot1 += hitEn;
      }
      for (unsigned int idet = 0; idet < (*t_DetIds3).size(); idet++) {
        unsigned int id = truncateId((*t_DetIds3)[idet], truncateFlag_, false);
        double cfac = corrFactor_->getCorr(id);
        if (cFactor_ != 0)
          cfac *= cFactor_->getCorr(t_Run, (*t_DetIds3)[idet]);
        double hitEn = cfac * (*t_HitEnergies3)[idet];
        Etot3 += hitEn;
      }
      ediff = (Etot3 - Etot1);
    }
    double fac = puFactor(-corrPU_, t_ieta, pmom, eHcal, ediff);
    eHcal *= fac;
  } else if (corrPU_ > 0) {
    eHcal = puFactorRho(corrPU_, t_ieta, t_rhoh, eHcal);
  }
}
