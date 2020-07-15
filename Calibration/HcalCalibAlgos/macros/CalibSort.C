//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibSort.C+g (for the tree "CalibTree")
//  CalibSort c1(fname, dirname, prefix, flag, mipCut);
//  c1.Loop("events.txt");
//  findDuplicate(infile, outfile, debug)
//
// .L CalibSort.C+g (for the tree "EventInfo")
//  CalibSortEvent c2(fname, dirname, prefix, append);
//  c2.Loop();
//  findDuplicateEvent(infile, outfile, debug)
//
//  This will prepare a list of dupliate entries from comombined data sets
//
//   where:
//
//   fname    (const char*)    = file name of the input ROOT tree
//                               or name of the file containing a list of
//                               file names of input ROOT trees
//   dirname  (std::string)    = name of the directory where Tree resides
//                               (default "HcalIsoTrkAnalyzer")
//   prefix  (std::string)     = String to be added to the name
//                               (usually a 4 character string; default="")
//   flag    (int)             = 2 digit integer (do) with specific control
//                               information (d = 0/1 for debug off/on;
//                               o = 0/1 for creating the output "events.txt"
//                               file in append/output mode. Default = 0
//   mipCut   (double)         = cut off on ECAL energy (default = 2.0)
//
//   infile   (std::string)    = name of the input file containing run, event,
//                               information (created by CalibSort)
//   outfile  (std::string)    = name of the file containing the entry numbers
//                               of duplicate events
//   debug    (bool)           = true/false for getting or not debug printouts
//                               (default = false)
//   append   (bool)           = flag used for opening o/p file. Default=false
//
//
// .L CalibSort.C+g
//  combine(fname1, fname2, fname0, outfile, debug)
//
//  Combines the 2 files created by isotrackApplyRegressor.py from
//  Models 1 (endcap) and 2 (barrel)
//
//   fname1/2 (std::string)    = file names for endcap/barrel correction
//                               factors
//   fname0   (std::string)    = output combined file
//   outfile  (std::string)    = root file name for storing histograms if
//                               a vlaid name is given (default = "")
//   debug    (int)            = verbosity flag (default = 1)
//
// .L CalibSort.C+g
//  plotCombine(infile, flag, drawStatBox, save)
//
//   infile       (const char*)= name of input ROOT file
//   flag         (int)        = indicates the source of the file.
//                               -1 : created by the method combine
//                               >=0 : created by CalibPlotCombine with
//                                     the same meaning as in that method
//   drawStatBox  (bool)       = flag to draw the statistics box (true)
//   save         (bool)       = save the plots (true)
//
// .L CalibSort.C+g (for the tree "CalibTree")
//  CalibPlotCombine c1(fname, corrFileName, dirname, flag);
//  c1.Loop()
//  c1.savePlot(histFileName,append)
//
//   fname        (const char*)= file name of the input ROOT tree
//                               or name of the file containing a list of
//                               file names of input ROOT trees
//   corrFileName (const char*)= name of the text file having the correction
//                               factors as a function of entry number
//   dirname      (std::string)= name of the directory where Tree resides
//                               ("HcalIsoTrkAnalyzer")
//   flag         (int)        = two digit number "to". o: specifies if
//                               charge isolation to be applied (1) or not
//                               (0); t: if all tracks to be included (0)
//                               or only with momentum 40-60 GeV (1)
//                               Default (10)
//   histFileName (std::string)= name of the histogram file
//   append       (bool)       = true/false if the histogram file to be opened
//                               in append/output mode
//
//////////////////////////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "CalibCorr.C"

struct record {
  record(int ser = 0, int ent = 0, int r = 0, int ev = 0, int ie = 0, double p = 0)
      : serial_(ser), entry_(ent), run_(r), event_(ev), ieta_(ie), p_(p) {}

  int serial_, entry_, run_, event_, ieta_;
  double p_;
};

struct recordLess {
  bool operator()(const record &a, const record &b) {
    return ((a.run_ < b.run_) || ((a.run_ == b.run_) && (a.event_ < b.event_)) ||
            ((a.run_ == b.run_) && (a.event_ == b.event_) && (a.ieta_ < b.ieta_)));
  }
};

struct recordEvent {
  recordEvent(unsigned int ser = 0, unsigned int ent = 0, unsigned int r = 0, unsigned int ev = 0)
      : serial_(ser), entry_(ent), run_(r), event_(ev) {}

  unsigned int serial_, entry_, run_, event_;
};

struct recordEventLess {
  bool operator()(const recordEvent &a, const recordEvent &b) {
    return ((a.run_ < b.run_) || ((a.run_ == b.run_) && (a.event_ < b.event_)));
  }
};

class CalibSort {
public:
  CalibSort(const char *fname,
            std::string dirname = "HcalIsoTrkAnalyzer",
            std::string prefix = "",
            bool allEvent = false,
            int flag = 0,
            double mipCut = 2.0);
  virtual ~CalibSort();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *);
  virtual void Loop(const char *);
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);

private:
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

  std::string fname_, dirnm_, prefix_;
  bool allEvent_;
  int flag_;
  double mipCut_;
};

CalibSort::CalibSort(const char *fname, std::string dirnm, std::string prefix, bool allEvent, int flag, double mipCut)
    : fname_(fname), dirnm_(dirnm), prefix_(prefix), allEvent_(allEvent), flag_(flag), mipCut_(mipCut) {
  char treeName[400];
  sprintf(treeName, "%s/CalibTree", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << std::endl;
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain);
  }
}

CalibSort::~CalibSort() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t CalibSort::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibSort::LoadTree(Long64_t entry) {
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

void CalibSort::Init(TChain *tree) {
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
}

Bool_t CalibSort::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibSort::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t CalibSort::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibSort::Loop(const char *outFile) {
  //   In a ROOT session, you can do:
  //      Root > .L CalibSort.C
  //      Root > CalibSort t
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

  std::ofstream fileout;
  if ((flag_ % 10) == 1) {
    fileout.open(outFile, std::ofstream::out);
    std::cout << "Opens " << outFile << " in output mode" << std::endl;
  } else {
    fileout.open(outFile, std::ofstream::app);
    std::cout << "Opens " << outFile << " in append mode" << std::endl;
  }
  if (!allEvent_)
    fileout << "Input file: " << fname_ << " Directory: " << dirnm_ << " Prefix: " << prefix_ << std::endl;
  Int_t runLow(99999999), runHigh(0);
  Long64_t nbytes(0), nb(0), good(0);
  Long64_t nentries = fChain->GetEntriesFast();
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (t_Run > 200000 && t_Run < 800000) {
      if (t_Run < runLow)
        runLow = t_Run;
      if (t_Run > runHigh)
        runHigh = t_Run;
    }
    double cut = (t_p > 20) ? 10.0 : 0.0;
    if ((flag_ / 10) % 10 > 0)
      std::cout << "Entry " << jentry << " p " << t_p << " Cuts " << t_qltyFlag << "|" << t_selectTk << "|"
                << (t_hmaxNearP < cut) << "|" << (t_eMipDR < mipCut_) << std::endl;
    if ((t_qltyFlag && t_selectTk && (t_hmaxNearP < cut) && (t_eMipDR < mipCut_)) || allEvent_) {
      good++;
      fileout << good << " " << jentry << " " << t_Run << " " << t_Event << " " << t_ieta << " " << t_p << std::endl;
    }
  }
  fileout.close();
  std::cout << "Writes " << good << " events in the file " << outFile << " from " << nentries
            << " entries in run range " << runLow << ":" << runHigh << std::endl;
}

class CalibSortEvent {
public:
  TChain *fChain;  //!pointer to the analyzed TTree/TChain
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

  CalibSortEvent(const char *fname, std::string dirname, std::string prefix = "", bool append = false);
  virtual ~CalibSortEvent();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *tree);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);

private:
  std::string fname_, dirnm_, prefix_;
  bool append_;
};

CalibSortEvent::CalibSortEvent(const char *fname, std::string dirnm, std::string prefix, bool append)
    : fname_(fname), dirnm_(dirnm), prefix_(prefix), append_(append) {
  char treeName[400];
  sprintf(treeName, "%s/EventInfo", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << std::endl;
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain);
  }
}

CalibSortEvent::~CalibSortEvent() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t CalibSortEvent::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibSortEvent::LoadTree(Long64_t entry) {
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

void CalibSortEvent::Init(TChain *tree) {
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
}

Bool_t CalibSortEvent::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibSortEvent::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t CalibSortEvent::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibSortEvent::Loop() {
  //   In a ROOT session, you can do:
  //      Root > .L CalibMonitor.C+g
  //      Root > CalibSortEvent t
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

  std::ofstream fileout;
  if (!append_) {
    fileout.open("runevents.txt", std::ofstream::out);
    std::cout << "Opens runevents.txt in output mode" << std::endl;
  } else {
    fileout.open("runevents.txt", std::ofstream::app);
    std::cout << "Opens runevents.txt in append mode" << std::endl;
  }
  fileout << "Input file: " << fname_ << " Directory: " << dirnm_ << " Prefix: " << prefix_ << std::endl;
  UInt_t runLow(99999999), runHigh(0);
  Long64_t nbytes(0), nb(0), good(0);
  Long64_t nentries = fChain->GetEntriesFast();
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (t_RunNo > 200000 && t_RunNo < 800000) {
      if (t_RunNo < runLow)
        runLow = t_RunNo;
      if (t_RunNo > runHigh)
        runHigh = t_RunNo;
    }
    good++;
    fileout << good << " " << jentry << " " << t_RunNo << " " << t_EventNo << std::endl;
  }
  fileout.close();
  std::cout << "Writes " << good << " events in the file events.txt from " << nentries << " entries in run range "
            << runLow << ":" << runHigh << std::endl;
}

void readRecords(std::string fname, std::vector<record> &records, bool debug) {
  records.clear();
  ifstream infile(fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      int ser, ent, r, ev, ie;
      double p;
      infile >> ser >> ent >> r >> ev >> ie >> p;
      if (!infile.good())
        break;
      record rec(ser, ent, r, ev, ie, p);
      records.push_back(rec);
    }
    infile.close();
  }
  std::cout << "Reads " << records.size() << " records from " << fname << std::endl;
  if (debug) {
    for (unsigned int k = 0; k < records.size(); ++k) {
      if (k % 100 == 0)
        std::cout << "[" << records[k].serial_ << ":" << records[k].entry_ << "] " << records[k].run_ << ":"
                  << records[k].event_ << " " << records[k].ieta_ << " " << records[k].p_ << std::endl;
    }
  }
}

void readMap(std::string fname, std::map<std::pair<int, int>, int> &records, bool debug) {
  records.clear();
  ifstream infile(fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      int ser, ent, r, ev, ie;
      double p;
      infile >> ser >> ent >> r >> ev >> ie >> p;
      if (!infile.good())
        break;
      std::pair<int, int> key(r, ev);
      if (records.find(key) == records.end())
        records[key] = ent;
    }
    infile.close();
  }
  std::cout << "Reads " << records.size() << " records from " << fname << std::endl;
  if (debug) {
    unsigned k(0);
    for (std::map<std::pair<int, int>, int>::iterator itr = records.begin(); itr != records.end(); ++itr, ++k) {
      if (k % 100 == 0)
        std::cout << "[" << k << "] " << itr->second << ":" << (itr->first).first << ":" << (itr->first).second << "\n";
    }
  }
}

void sort(std::vector<record> &records, bool debug) {
  // Use std::sort
  std::sort(records.begin(), records.end(), recordLess());
  if (debug) {
    for (unsigned int k = 0; k < records.size(); ++k) {
      std::cout << "[" << k << ":" << records[k].serial_ << ":" << records[k].entry_ << "] " << records[k].run_ << ":"
                << records[k].event_ << " " << records[k].ieta_ << " " << records[k].p_ << std::endl;
    }
  }
}

void duplicate(std::string fname, std::vector<record> &records, bool debug) {
  std::ofstream file;
  file.open(fname.c_str(), std::ofstream::out);
  std::cout << "List of entry names of duplicate events" << std::endl;
  int duplicate(0), dupl40(0);
  for (unsigned int k = 1; k < records.size(); ++k) {
    if ((records[k].run_ == records[k - 1].run_) && (records[k].event_ == records[k - 1].event_) &&
        (records[k].ieta_ == records[k - 1].ieta_) && (fabs(records[k].p_ - records[k - 1].p_) < 0.0001)) {
      // This is a duplicate event - reject the one with larger serial #
      if (records[k].entry_ < records[k - 1].entry_) {
        record swap = records[k - 1];
        records[k - 1] = records[k];
        records[k] = swap;
      }
      if (debug) {
        std::cout << "Serial " << records[k - 1].serial_ << ":" << records[k].serial_ << " Entry "
                  << records[k - 1].entry_ << ":" << records[k].entry_ << " Run " << records[k - 1].run_ << ":"
                  << records[k].run_ << " Event " << records[k - 1].event_ << " " << records[k].event_ << " Eta "
                  << records[k - 1].ieta_ << " " << records[k].ieta_ << " p " << records[k - 1].p_ << ":"
                  << records[k].p_ << std::endl;
      }
      file << records[k].entry_ << std::endl;
      duplicate++;
      if (records[k].p_ >= 40.0 && records[k].p_ <= 60.0)
        dupl40++;
    }
  }
  file.close();
  std::cout << "Total # of duplcate events " << duplicate << " (" << dupl40 << " with p 40:60)" << std::endl;
}

void findDuplicate(std::string infile, std::string outfile, bool debug = false) {
  std::vector<record> records;
  readRecords(infile, records, debug);
  sort(records, debug);
  duplicate(outfile, records, debug);
}

void findCommon(std::string infile1, std::string infile2, std::string infile3, std::string outfile, bool debug = false) {
  std::map<std::pair<int, int>, int> map1, map2, map3;
  readMap(infile1, map1, debug);
  readMap(infile2, map2, debug);
  readMap(infile3, map3, debug);
  bool check3 = (map3.size() > 0);
  std::ofstream file;
  file.open(outfile.c_str(), std::ofstream::out);
  unsigned int k(0), good(0);
  for (std::map<std::pair<int, int>, int>::iterator itr = map1.begin(); itr != map1.end(); ++itr, ++k) {
    std::pair<int, int> key = itr->first;
    bool ok = (map2.find(key) != map2.end());
    if (ok && check3)
      ok = (map3.find(key) != map3.end());
    if (debug && k % 100 == 0)
      std::cout << "[" << k << "] Run " << key.first << " Event " << key.second << " Flag " << ok << std::endl;
    if (ok) {
      ++good;
      file << key.first << "   " << key.second << std::endl;
    }
  }
  file.close();
  std::cout << "Total # of common events " << good << " written to o/p file " << outfile << std::endl;
}

void readRecordEvents(std::string fname, std::vector<recordEvent> &records, bool debug) {
  records.clear();
  ifstream infile(fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      unsigned int ser, ent, r, ev;
      infile >> ser >> ent >> r >> ev;
      if (!infile.good())
        break;
      recordEvent rec(ser, ent, r, ev);
      records.push_back(rec);
    }
    infile.close();
  }
  std::cout << "Reads " << records.size() << " records from " << fname << std::endl;
  if (debug) {
    for (unsigned int k = 0; k < records.size(); ++k) {
      if (k % 100 == 0)
        std::cout << "[" << records[k].serial_ << ":" << records[k].entry_ << "] " << records[k].run_ << ":"
                  << records[k].event_ << std::endl;
    }
  }
}

void sortEvent(std::vector<recordEvent> &records, bool debug) {
  // Use std::sort
  std::sort(records.begin(), records.end(), recordEventLess());
  if (debug) {
    for (unsigned int k = 0; k < records.size(); ++k) {
      std::cout << "[" << k << ":" << records[k].serial_ << ":" << records[k].entry_ << "] " << records[k].run_ << ":"
                << records[k].event_ << std::endl;
    }
  }
}

void duplicateEvent(std::string fname, std::vector<recordEvent> &records, bool debug) {
  std::ofstream file;
  file.open(fname.c_str(), std::ofstream::out);
  std::cout << "List of entry names of duplicate events" << std::endl;
  int duplicate(0);
  for (unsigned int k = 1; k < records.size(); ++k) {
    if ((records[k].run_ == records[k - 1].run_) && (records[k].event_ == records[k - 1].event_)) {
      // This is a duplicate event - reject the one with larger serial #
      if (records[k].entry_ < records[k - 1].entry_) {
        recordEvent swap = records[k - 1];
        records[k - 1] = records[k];
        records[k] = swap;
      }
      if (debug) {
        std::cout << "Serial " << records[k - 1].serial_ << ":" << records[k].serial_ << " Entry "
                  << records[k - 1].entry_ << ":" << records[k].entry_ << " Run " << records[k - 1].run_ << ":"
                  << records[k].run_ << " Event " << records[k - 1].event_ << " " << records[k].event_ << std::endl;
      }
      file << records[k].entry_ << std::endl;
      duplicate++;
    }
  }
  file.close();
  std::cout << "Total # of duplcate events " << duplicate << std::endl;
}

void findDuplicateEvent(std::string infile, std::string outfile, bool debug = false) {
  std::vector<recordEvent> records;
  readRecordEvents(infile, records, debug);
  sortEvent(records, debug);
  duplicateEvent(outfile, records, debug);
}

void combine(const std::string fname1,
             const std::string fname2,
             const std::string fname,
             const std::string outfile = "",
             int debug = 0) {
  ifstream infile1(fname1.c_str());
  ifstream infile2(fname2.c_str());
  if ((!infile1.is_open()) || (!infile2.is_open())) {
    std::cout << "Cannot open " << fname1 << " or " << fname2 << std::endl;
  } else {
    std::ofstream file;
    file.open(fname.c_str(), std::ofstream::out);
    TH1D *hist0 = new TH1D("W0", "Correction Factor (All)", 100, 0.0, 10.0);
    TH1D *hist1 = new TH1D("W1", "Correction Factor (Barrel)", 100, 0.0, 10.0);
    TH1D *hist2 = new TH1D("W2", "Correction Factor (Endcap)", 100, 0.0, 10.0);
    TProfile *prof = new TProfile("P", "Correction vs i#eta", 60, -30, 30, 0, 100, "");
    unsigned int kount(0), kout(0);
    double wmaxb(0), wmaxe(0);
    while (1) {
      Long64_t entry1, entry2;
      int ieta1, ieta2;
      double wt1, wt2;
      infile1 >> entry1 >> ieta1 >> wt1;
      infile2 >> entry2 >> ieta2 >> wt2;
      if ((!infile1.good()) || (!infile2.good()))
        break;
      ++kount;
      if (debug > 1) {
        std::cout << kount << " Enrty No. " << entry1 << ":" << entry2 << " eta " << ieta1 << ":" << ieta2 << " wt "
                  << wt1 << ":" << wt2 << std::endl;
      }
      if (entry1 == entry2) {
        int ieta = fabs(ieta1);
        double wt = (ieta >= 16) ? wt1 : wt2;
        file << std::setw(8) << entry1 << " " << std::setw(12) << std::setprecision(8) << wt << std::endl;
        hist0->Fill(wt);
        if (ieta >= 16) {
          hist2->Fill(wt);
          if (wt > wmaxe)
            wmaxe = wt;
        } else {
          hist1->Fill(wt);
          if (wt > wmaxb)
            wmaxb = wt;
        }
        prof->Fill(ieta1, wt);
        ++kout;
      } else if (debug > 0) {
        std::cout << kount << " Entry " << entry1 << ":" << entry2 << " eta " << ieta1 << ":" << ieta2 << " wt " << wt1
                  << ":" << wt2 << " mismatch" << std::endl;
      }
    }
    infile1.close();
    infile2.close();
    file.close();
    std::cout << "Writes " << kout << " entries to " << fname << " from " << kount << " events from " << fname1
              << " and " << fname2 << " maximum correction factor " << wmaxb << " (Barrel) " << wmaxe << " (Endcap)"
              << std::endl;
    if (outfile != "") {
      TFile *theFile = new TFile(outfile.c_str(), "RECREATE");
      theFile->cd();
      hist0->Write();
      hist1->Write();
      hist2->Write();
      prof->Write();
      theFile->Close();
    }
  }
}

void plotCombine(const char *infile, int flag = -1, bool drawStatBox = true, bool save = true) {
  int flag1 = (flag >= 0 ? (flag / 10) % 10 : 0);
  int flag2 = (flag >= 0 ? (flag % 10) : 0);
  std::string name[4] = {"W0", "W1", "W2", "P"};
  std::string xtitl[4] = {
      "Correction Factor (All)", "Correction Factor (Barrel)", "Correction Factor (Endcap)", "i#eta"};
  std::string ytitl[4] = {"Track", "Track", "Track", "Correction Factor"};
  char title[100];
  std::string mom[2] = {"all momentum", "p = 40:60 GeV"};
  std::string iso[2] = {"loose", "tight"};
  if (flag < 0) {
    sprintf(title, "All tracks");
  } else {
    sprintf(title, "Good tracks of %s with %s isolation", mom[flag1].c_str(), iso[flag2].c_str());
  }

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(0);
  TFile *file = new TFile(infile);
  for (int k = 0; k < 4; ++k) {
    char nameh[40], namep[50];
    if (flag >= 0)
      sprintf(nameh, "%s%d%d", name[k].c_str(), flag1, flag2);
    else
      sprintf(nameh, "%s", name[k].c_str());
    sprintf(namep, "c_%s", nameh);
    TCanvas *pad(nullptr);
    if (k == 3) {
      TProfile *hist = (TProfile *)file->FindObjectAny(nameh);
      if (hist != nullptr) {
        pad = new TCanvas(namep, namep, 700, 500);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        hist->GetXaxis()->SetTitleSize(0.04);
        hist->GetXaxis()->SetTitle(xtitl[k].c_str());
        hist->GetYaxis()->SetTitle(ytitl[k].c_str());
        hist->GetYaxis()->SetLabelOffset(0.005);
        hist->GetYaxis()->SetTitleSize(0.04);
        hist->GetYaxis()->SetLabelSize(0.035);
        hist->GetYaxis()->SetTitleOffset(1.10);
        hist->SetMarkerStyle(20);
        hist->Draw();
        pad->Update();
        TPaveStats *st1 = (TPaveStats *)hist->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr && drawStatBox) {
          st1->SetY1NDC(0.70);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
      }
    } else {
      TH1D *hist = (TH1D *)file->FindObjectAny(nameh);
      if (hist != nullptr) {
        pad = new TCanvas(namep, namep, 700, 500);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        pad->SetLogy();
        hist->GetXaxis()->SetTitleSize(0.04);
        hist->GetXaxis()->SetTitle(xtitl[k].c_str());
        hist->GetYaxis()->SetTitle(ytitl[k].c_str());
        hist->GetYaxis()->SetLabelOffset(0.005);
        hist->GetYaxis()->SetTitleSize(0.04);
        hist->GetYaxis()->SetLabelSize(0.035);
        hist->GetYaxis()->SetTitleOffset(1.10);
        hist->SetMarkerStyle(20);
        hist->Draw();
        pad->Update();
        TPaveStats *st1 = (TPaveStats *)hist->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr && drawStatBox) {
          st1->SetY1NDC(0.70);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
      }
      TPaveText *txt0 = new TPaveText(0.10, 0.91, 0.90, 0.96, "blNDC");
      txt0->SetFillColor(0);
      txt0->AddText(title);
      txt0->Draw("same");
      pad->Update();
    }
    if (pad != nullptr) {
      pad->Modified();
      pad->Update();
      if (save) {
        sprintf(namep, "%s.pdf", pad->GetName());
        pad->Print(namep);
      }
    }
  }
}

class CalibPlotCombine {
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

  CalibPlotCombine(const char *fname,
                   const char *corrFileName,
                   const std::string dirname = "HcalIsoTrkAnalyzer",
                   int flag = 10);
  virtual ~CalibPlotCombine();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  bool goodTrack(double eHcal, double cut);
  void savePlot(const std::string &theName, bool append = false);

private:
  CalibCorr *cFactor_;
  int flag_, ifDepth_;
  bool selectP_, tightSelect_;
  TH1D *hist0_, *hist1_, *hist2_;
  TProfile *prof_;
};

CalibPlotCombine::CalibPlotCombine(const char *fname, const char *corrFileName, const std::string dirnm, int flag)
    : cFactor_(nullptr), flag_(flag), ifDepth_(3) {
  selectP_ = (((flag_ / 10) % 10) > 0);
  tightSelect_ = ((flag_ % 10) > 0);
  flag_ = 0;
  if (selectP_)
    flag_ += 10;
  if (tightSelect_)
    ++flag_;
  std::cout << "Selection on momentum range: " << selectP_ << std::endl
            << "Tight isolation flag:        " << tightSelect_ << std::endl
            << "Flag:                        " << flag_ << std::endl;
  char treeName[400];
  sprintf(treeName, "%s/CalibTree", dirnm.c_str());
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << fname << std::endl;
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain);
  }
  cFactor_ = new CalibCorr(corrFileName, ifDepth_, false);
}

CalibPlotCombine::~CalibPlotCombine() {
  delete cFactor_;
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t CalibPlotCombine::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibPlotCombine::LoadTree(Long64_t entry) {
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

void CalibPlotCombine::Init(TChain *tree) {
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

  // Book the histograms
  char name[20], title[100];
  std::string mom[2] = {"all momentum", "p = 40:60 GeV"};
  std::string iso[2] = {"loose", "tight"};
  std::string rng[3] = {"All", "Barrel", "Endcap"};
  int flag1 = (flag_ / 10) % 10;
  int flag2 = (flag_ % 10);
  sprintf(name, "W0%d%d", flag1, flag2);
  sprintf(title,
          "Correction Factor (%s) for tracks with %s and %s isolation",
          rng[0].c_str(),
          mom[flag1].c_str(),
          iso[flag2].c_str());
  hist0_ = new TH1D(name, title, 100, 0.0, 10.0);
  sprintf(name, "W1%d%d", flag1, flag2);
  sprintf(title,
          "Correction Factor (%s) for tracks with %s and %s isolation",
          rng[1].c_str(),
          mom[flag1].c_str(),
          iso[flag2].c_str());
  hist1_ = new TH1D(name, title, 100, 0.0, 10.0);
  sprintf(name, "W2%d%d", flag1, flag2);
  sprintf(title,
          "Correction Factor (%s) for tracks with %s and %s isolation",
          rng[2].c_str(),
          mom[flag1].c_str(),
          iso[flag2].c_str());
  hist2_ = new TH1D(name, title, 100, 0.0, 10.0);
  sprintf(name, "P%d%d", flag1, flag2);
  sprintf(
      title, "Correction Factor vs i#eta for tracks with %s and %s isolation", mom[flag1].c_str(), iso[flag2].c_str());
  prof_ = new TProfile(name, title, 60, -30, 30, 0, 100, "");
}

Bool_t CalibPlotCombine::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibPlotCombine::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t CalibPlotCombine::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibPlotCombine::Loop() {
  //   In a ROOT session, you can do:
  //      Root > .L CalibPlotCombine.C
  //      Root > CalibPlotCombine t
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
  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << std::endl;
  Long64_t nbytes(0), nb(0);
  double wminb(99999999), wmaxb(0), wmine(99999999), wmaxe(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    if (jentry % 1000000 == 0)
      std::cout << "Entry " << jentry << " Run " << t_Run << " Event " << t_Event << std::endl;
    double cut = (t_p > 20) ? (tightSelect_ ? 2.0 : 10.0) : 0.0;
    double eHcal(t_eHcal);
    if (goodTrack(eHcal, cut)) {
      if ((!selectP_) || ((t_p > 40.0) && (t_p < 60.0))) {
        int ieta = fabs(t_ieta);
        double wt = cFactor_->getTrueCorr(ientry);
        hist0_->Fill(wt);
        if (ieta >= 16) {
          hist2_->Fill(wt);
          wmaxe = std::max(wmaxe, wt);
          wmine = std::min(wmine, wt);
        } else {
          hist1_->Fill(wt);
          wmaxb = std::max(wmaxb, wt);
          wminb = std::max(wminb, wt);
        }
        prof_->Fill(t_ieta, wt);
      }
    }
  }
  std::cout << "Minimum and maximum correction factors " << wmine << ":" << wmaxb << " (Barrel) " << wmine << ":"
            << wmaxe << " (Endcap)" << std::endl;
}

bool CalibPlotCombine::goodTrack(double eHcal, double cut) {
  return ((t_qltyFlag) && t_selectTk && (t_hmaxNearP < cut) && (t_eMipDR < 1.0) && (eHcal > 0.001));
}

void CalibPlotCombine::savePlot(const std::string &theName, bool append) {
  TFile *theFile = (append ? (new TFile(theName.c_str(), "UPDATE")) : (new TFile(theName.c_str(), "RECREATE")));
  theFile->cd();
  TH1D *hist;
  hist = (TH1D *)(hist0_->Clone());
  hist->Write();
  hist = (TH1D *)(hist1_->Clone());
  hist->Write();
  hist = (TH1D *)(hist2_->Clone());
  hist->Write();
  TProfile *prof = (TProfile *)(prof_->Clone());
  prof->Write();
  std::cout << "All done" << std::endl;
  theFile->Close();
}
