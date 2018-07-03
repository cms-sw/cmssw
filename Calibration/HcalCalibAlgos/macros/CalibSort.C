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
//  This will prepare a list of dupliate entries from combined data sets
//
//   where:
// 
//   fname   (std::string)     = file name of the input ROOT tree
//   dirname (std::string)     = name of the directory where Tree resides
//                               (default "HcalIsoTrkAnalyzer")
//   prefix (std::string)      = String to be added to the name
//                               (usually a 4 character string; default="")
//   flag   (int)              = 2 digit integer (do) with specific control
//                               information (d = 0/1 for debug off/on;
//                               o = 0/1 for creating the output "events.txt"
//                               file in append/output mode. Default = 0
//   mipCut  (double)          = cut off on ECAL energy (default = 2.0)
//
//   infile  (std::string)     = name of the input file containing run, event,
//                               information (created by CalibSort)
//   outfile (std::string)     = name of the file containing the entry numbers
//                               of duplicate events
//   debug   (bool)            = true/false for getting or not debug printouts
//                               (default = false)
//   append  (bool)            = flag used for opening o/p file. Default=false
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

struct record {
  record(int ser=0, int ent=0, int r=0, int ev=0, int ie=0, double p=0) :
    serial_(ser), entry_(ent), run_(r), event_(ev), ieta_(ie), p_(p) {}

  int    serial_, entry_, run_, event_, ieta_;
  double p_;
};

struct recordLess {
  bool operator() (const record& a, const record& b) {
    return ((a.run_ < b.run_) || 
	    ((a.run_ == b.run_) && (a.event_ <  b.event_)) ||
	    ((a.run_ == b.run_) && (a.event_ == b.event_) && 
	     (a.ieta_ < b.ieta_)));
  }
};

struct recordEvent {
  recordEvent(unsigned int ser=0, unsigned int ent=0, unsigned int r=0,
	      unsigned int ev=0) : serial_(ser), entry_(ent), run_(r),
				   event_(ev) {}

  unsigned int serial_, entry_, run_, event_;
};

struct recordEventLess {
  bool operator() (const recordEvent& a, const recordEvent& b) {
    return ((a.run_ < b.run_) || 
	    ((a.run_ == b.run_) && (a.event_ <  b.event_)));
  }
};

class CalibSort {
public :
  CalibSort(std::string fname, std::string dirname="HcalIsoTrkAnalyzer",
	    std::string prefix="", bool allEvent=false, int flag=0, 
	    double mipCut=2.0);
  virtual ~CalibSort();
  virtual Int_t              Cut(Long64_t entry);
  virtual Int_t              GetEntry(Long64_t entry);
  virtual Long64_t           LoadTree(Long64_t entry);
  virtual void               Init(TTree*);
  virtual void               Loop(const char*);
  virtual Bool_t             Notify();
  virtual void               Show(Long64_t entry = -1);
private:

  TTree                     *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t                      fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t                      t_Run;
  Int_t                      t_Event;
  Int_t                      t_DataType;
  Int_t                      t_ieta;
  Int_t                      t_iphi;
  Double_t                   t_EventWeight;
  Int_t                      t_nVtx;
  Int_t                      t_nTrk;
  Int_t                      t_goodPV;
  Double_t                   t_l1pt;
  Double_t                   t_l1eta;
  Double_t                   t_l1phi;
  Double_t                   t_l3pt;
  Double_t                   t_l3eta;
  Double_t                   t_l3phi;
  Double_t                   t_p;
  Double_t                   t_pt;
  Double_t                   t_phi;
  Double_t                   t_mindR1;
  Double_t                   t_mindR2;
  Double_t                   t_eMipDR;
  Double_t                   t_eHcal;
  Double_t                   t_eHcal10;
  Double_t                   t_eHcal30;
  Double_t                   t_hmaxNearP;
  Double_t                   t_rhoh;
  Bool_t                     t_selectTk;
  Bool_t                     t_qltyFlag;
  Bool_t                     t_qltyMissFlag;
  Bool_t                     t_qltyPVFlag;  
  Double_t                   t_gentrackP;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies;
  std::vector<bool>         *t_trgbits;
  std::vector<unsigned int> *t_DetIds1;
  std::vector<unsigned int> *t_DetIds3;
  std::vector<double>       *t_HitEnergies1;
  std::vector<double>       *t_HitEnergies3;

  // List of branches
  TBranch                   *b_t_Run;           //!
  TBranch                   *b_t_Event;         //!
  TBranch                   *b_t_DataType;      //!
  TBranch                   *b_t_ieta;          //!
  TBranch                   *b_t_iphi;          //!
  TBranch                   *b_t_EventWeight;   //!
  TBranch                   *b_t_nVtx;          //!
  TBranch                   *b_t_nTrk;          //!
  TBranch                   *b_t_goodPV;        //!
  TBranch                   *b_t_l1pt;          //!
  TBranch                   *b_t_l1eta;         //!
  TBranch                   *b_t_l1phi;         //!
  TBranch                   *b_t_l3pt;          //!
  TBranch                   *b_t_l3eta;         //!
  TBranch                   *b_t_l3phi;         //!
  TBranch                   *b_t_p;             //!
  TBranch                   *b_t_pt;            //!
  TBranch                   *b_t_phi;           //!
  TBranch                   *b_t_mindR1;        //!
  TBranch                   *b_t_mindR2;        //!
  TBranch                   *b_t_eMipDR;        //!
  TBranch                   *b_t_eHcal;         //!
  TBranch                   *b_t_eHcal10;       //!
  TBranch                   *b_t_eHcal30;       //!
  TBranch                   *b_t_hmaxNearP;     //!
  TBranch                   *b_t_rhoh;          //!
  TBranch                   *b_t_selectTk;      //!
  TBranch                   *b_t_qltyFlag;      //!
  TBranch                   *b_t_qltyMissFlag;  //!
  TBranch                   *b_t_qltyPVFlag;    //!
  TBranch                   *b_t_gentrackP;     //!
  TBranch                   *b_t_DetIds;        //!
  TBranch                   *b_t_HitEnergies;   //!
  TBranch                   *b_t_trgbits;       //!
  TBranch                   *b_t_DetIds1;       //!
  TBranch                   *b_t_DetIds3;       //!
  TBranch                   *b_t_HitEnergies1;  //!
  TBranch                   *b_t_HitEnergies3;  //!

  std::string               fname_, dirnm_, prefix_;
  bool                      allEvent_;
  int                       flag_;
  double                    mipCut_;
};

CalibSort::CalibSort(std::string fname, std::string dirnm, 
		     std::string prefix, bool allEvent, int flag, 
		     double mipCut) : fname_(fname), dirnm_(dirnm), 
				      prefix_(prefix), allEvent_(allEvent),
				      flag_(flag), mipCut_(mipCut) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree
  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir 
	    << std::endl;
  TTree      *tree = (TTree*)dir->Get("CalibTree");
  std::cout << "CalibTree " << tree << std::endl;
  Init(tree);
}

CalibSort::~CalibSort() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t CalibSort::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibSort::LoadTree(Long64_t entry) {
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

void CalibSort::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).
  
  // Set object pointer
  t_DetIds       = 0;
  t_DetIds1      = 0;
  t_DetIds3      = 0;
  t_HitEnergies  = 0;
  t_HitEnergies1 = 0;
  t_HitEnergies3 = 0;
  t_trgbits      = 0;
  // Set branch addresses and branch pointers
  fChain   = tree;
  fCurrent = -1;
  if (!tree) return;
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
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t CalibSort::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibSort::Loop(const char* outFile) {
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
  if (fChain == 0) return;

  std::ofstream fileout;
  if ((flag_%10)==1) {
    fileout.open(outFile, std::ofstream::out);
    std::cout << "Opens " << outFile << " in output mode" << std::endl;
  } else {
    fileout.open(outFile, std::ofstream::app);
    std::cout << "Opens " << outFile << " in append mode" << std::endl;
  }
  if (!allEvent_)
    fileout << "Input file: " << fname_ << " Directory: " << dirnm_ 
	    << " Prefix: " << prefix_ << std::endl;
  Int_t runLow(99999999), runHigh(0);
  Long64_t nbytes(0), nb(0), good(0);
  Long64_t nentries = fChain->GetEntriesFast();
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if (t_Run > 200000 && t_Run < 800000) {
      if (t_Run < runLow)  runLow  = t_Run;
      if (t_Run > runHigh) runHigh = t_Run;
    }
    double cut = (t_p > 20) ? 10.0 : 0.0;
    if ((flag_/10)%10 > 0) 
      std::cout << "Entry " << jentry << " p " << t_p << " Cuts " << t_qltyFlag
		<< "|" << t_selectTk << "|" << (t_hmaxNearP < cut) << "|" 
		<< (t_eMipDR < mipCut_) << std::endl;
    if ((t_qltyFlag && t_selectTk && (t_hmaxNearP<cut) && (t_eMipDR<mipCut_)) ||
	allEvent_) {
      good++;
      fileout << good << " " << jentry << " " << t_Run  << " " << t_Event 
	      << " " << t_ieta << " " << t_p << std::endl;
    }
  }
  fileout.close();
  std::cout << "Writes " << good << " events in the file " << outFile 
	    << " from " << nentries << " entries in run range " << runLow 
	    << ":" << runHigh << std::endl;
}

class CalibSortEvent {
public :
  TTree                     *fChain;   //!pointer to the analyzed TTree/TChain
  Int_t                      fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  UInt_t                     t_RunNo;
  UInt_t                     t_EventNo;
  Int_t                      t_Tracks;
  Int_t                      t_TracksProp;
  Int_t                      t_TracksSaved;
  Int_t                      t_TracksLoose;
  Int_t                      t_TracksTight;
  Int_t                      t_allvertex;
  Bool_t                     t_TrigPass;
  Bool_t                     t_TrigPassSel;
  Bool_t                     t_L1Bit;
  std::vector<Bool_t>       *t_hltbits;
  std::vector<int>          *t_ietaAll;
  std::vector<int>          *t_ietaGood;
  std::vector<int>          *t_trackType;

  // List of branches
  TBranch                   *b_t_RunNo;         //!
  TBranch                   *b_t_EventNo;       //!
  TBranch                   *b_t_Tracks;        //!
  TBranch                   *b_t_TracksProp;    //!
  TBranch                   *b_t_TracksSaved;   //!
  TBranch                   *b_t_TracksLoose;   //!
  TBranch                   *b_t_TracksTight;   //!
  TBranch                   *b_t_allvertex;     //!
  TBranch                   *b_t_TrigPass;      //!
  TBranch                   *b_t_TrigPassSel;   //!
  TBranch                   *b_t_L1Bit;         //!
  TBranch                   *b_t_hltbits;       //!
  TBranch                   *b_t_ietaAll;       //!
  TBranch                   *b_t_ietaGood;      //!
  TBranch                   *b_t_trackType;     //!

  CalibSortEvent(std::string fname, std::string dirname, std::string prefix="",
		 bool append=false);
  virtual ~CalibSortEvent();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

private:
  std::string      fname_, dirnm_, prefix_;
  bool             append_;
};

CalibSortEvent::CalibSortEvent(std::string fname, std::string dirnm,
			       std::string prefix, bool append) :
  fname_(fname), dirnm_(dirnm), prefix_(prefix), append_(append) {

  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir 
	    << std::endl;
  TTree      *tree = (TTree*)dir->Get("EventInfo");
  std::cout << "CalibSortEvent " << tree << std::endl;
  Init(tree);
}

CalibSortEvent::~CalibSortEvent() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t CalibSortEvent::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibSortEvent::LoadTree(Long64_t entry) {
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

void CalibSortEvent::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set branch addresses and branch pointers
  // Set object pointer
  t_hltbits      = 0;
  t_ietaAll      = 0;
  t_ietaGood     = 0;
  t_trackType    = 0;
  t_L1Bit        = false;
  fChain         = tree;
  fCurrent       = -1;
  if (!tree) return;
  fChain->SetMakeClass(1);
  fChain->SetBranchAddress("t_RunNo",       &t_RunNo,       &b_t_RunNo);
  fChain->SetBranchAddress("t_EventNo",     &t_EventNo,     &b_t_EventNo);
  fChain->SetBranchAddress("t_Tracks",      &t_Tracks,      &b_t_Tracks);
  fChain->SetBranchAddress("t_TracksProp",  &t_TracksProp,  &b_t_TracksProp);
  fChain->SetBranchAddress("t_TracksSaved", &t_TracksSaved, &b_t_TracksSaved);
  fChain->SetBranchAddress("t_TracksLoose", &t_TracksLoose, &b_t_TracksLoose);
  fChain->SetBranchAddress("t_TracksTight", &t_TracksTight, &b_t_TracksTight);
  fChain->SetBranchAddress("t_allvertex",   &t_allvertex,   &b_t_allvertex);
  fChain->SetBranchAddress("t_TrigPass",    &t_TrigPass,    &b_t_TrigPass);
  fChain->SetBranchAddress("t_TrigPassSel", &t_TrigPassSel, &b_t_TrigPassSel);
  fChain->SetBranchAddress("t_L1Bit",       &t_L1Bit,       &b_t_L1Bit);
  fChain->SetBranchAddress("t_hltbits",     &t_hltbits,     &b_t_hltbits);
  fChain->SetBranchAddress("t_ietaAll",     &t_ietaAll,     &b_t_ietaAll);
  fChain->SetBranchAddress("t_ietaGood",    &t_ietaGood,    &b_t_ietaGood);
  fChain->SetBranchAddress("t_trackType",   &t_trackType,   &b_t_trackType);
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
  if (!fChain) return;
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
  if (fChain == 0) return;

  std::ofstream fileout;
  if (!append_) {
    fileout.open("runevents.txt", std::ofstream::out);
    std::cout << "Opens runevents.txt in output mode" << std::endl;
  } else {
    fileout.open("runevents.txt", std::ofstream::app);
    std::cout << "Opens runevents.txt in append mode" << std::endl;
  }
  fileout << "Input file: " << fname_ << " Directory: " << dirnm_ 
	  << " Prefix: " << prefix_ << std::endl;
  UInt_t   runLow(99999999), runHigh(0);
  Long64_t nbytes(0), nb(0), good(0);
  Long64_t nentries = fChain->GetEntriesFast();
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if (t_RunNo > 200000 && t_RunNo < 800000) {
      if (t_RunNo < runLow)  runLow  = t_RunNo;
      if (t_RunNo > runHigh) runHigh = t_RunNo;
    }
    good++;
    fileout << good << " " << jentry << " " << t_RunNo  << " " << t_EventNo 
	    << std::endl;
  }
  fileout.close();
  std::cout << "Writes " << good << " events in the file events.txt from "
	    << nentries << " entries in run range " << runLow << ":"
 	    << runHigh << std::endl;
}

void readRecords(std::string fname, std::vector<record>& records, bool debug) {
  records.clear();
  ifstream infile (fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      int ser, ent, r, ev, ie;
      double p;
      infile >> ser >> ent >> r >> ev >> ie >> p;
      if (!infile.good()) break;
      record rec(ser,ent,r,ev,ie,p);
      records.push_back(rec);
    }
    infile.close();
  }
  std::cout << "Reads " << records.size() << " records from " << fname 
	    << std::endl;
  if (debug) {
    for (unsigned int k=0; k<records.size(); ++k) {
      if (k%100 == 0) 
	std::cout << "[" << records[k].serial_ << ":" << records[k].entry_ 
		  << "] " << records[k].run_ << ":" << records[k].event_ << " "
		  << records[k].ieta_ << " " << records[k].p_  << std::endl;
    }
  }
}

void readMap(std::string fname, std::map<std::pair<int,int>,int>& records, 
	     bool debug) {
  records.clear();
  ifstream infile (fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      int ser, ent, r, ev, ie;
      double p;
      infile >> ser >> ent >> r >> ev >> ie >> p;
      if (!infile.good()) break;
      std::pair<int,int> key(r,ev);
      if (records.find(key) == records.end()) records[key] = ent;
    }
    infile.close();
  }
  std::cout << "Reads " << records.size() << " records from " << fname 
	    << std::endl;
  if (debug) {
    unsigned k(0);
    for (std::map<std::pair<int,int>,int>::iterator itr=records.begin();
	 itr != records.end(); ++itr,++k) {
      if (k%100 == 0) 
	std::cout << "[" << k << "] " << itr->second << ":" 
		  << (itr->first).first << ":" << (itr->first).second << "\n";
    }
  }
}

void sort(std::vector<record>& records, bool debug) {
  // Use std::sort
  std::sort(records.begin(), records.end(), recordLess());
  if (debug) {
    for (unsigned int k=0; k<records.size(); ++k) {
      std::cout << "[" << k << ":" << records[k].serial_ << ":" 
		<< records[k].entry_ << "] " << records[k].run_ << ":" 
		<< records[k].event_ << " " << records[k].ieta_ << " " 
		<< records[k].p_ << std::endl;
    }
  }
}


void duplicate (std::string fname, std::vector<record>& records, bool debug) {
  std::ofstream file;
  file.open(fname.c_str(), std::ofstream::out);
  std::cout << "List of entry names of duplicate events" << std::endl;
  int duplicate(0), dupl40(0);
  for (unsigned int k=1; k<records.size(); ++k) {
    if ((records[k].run_ == records[k-1].run_) &&
	(records[k].event_ == records[k-1].event_) &&
	(records[k].ieta_ == records[k-1].ieta_) &&
	(fabs(records[k].p_-records[k-1].p_) < 0.0001)) {
      // This is a duplicate event - reject the one with larger serial #
      if (records[k].entry_ < records[k-1].entry_) {
	record swap = records[k-1];
	records[k-1]= records[k];
	records[k]  = swap;
      }
      if (debug) {
	std::cout << "Serial " << records[k-1].serial_ << ":"  
		  << records[k].serial_ << " Entry "  
		  << records[k-1].entry_ << ":" << records[k].entry_ << " Run "
		  << records[k-1].run_ << ":"  << records[k].run_ << " Event "
		  << records[k-1].event_ << " " << records[k].event_ << " Eta "
		  << records[k-1].ieta_ << " " << records[k].ieta_ << " p "
		  << records[k-1].p_ << ":" << records[k].p_ << std::endl;
      }
      file << records[k].entry_ << std::endl;
      duplicate++;
      if (records[k].p_ >= 40.0 && records[k].p_ <= 60.0) dupl40++;
    }
  }
  file.close();
  std::cout << "Total # of duplcate events " << duplicate << " (" << dupl40 
	    << " with p 40:60)" << std::endl;
}

void findDuplicate(std::string infile, std::string outfile, bool debug=false) {

  std::vector<record> records;
  readRecords(infile, records, debug);
  sort(records,debug);
  duplicate(outfile, records, debug);
}

void findCommon(std::string infile1, std::string infile2, std::string infile3,
		std::string outfile, bool debug=false) {
  std::map<std::pair<int,int>,int> map1, map2, map3;
  readMap(infile1,map1,debug);
  readMap(infile2,map2,debug);
  readMap(infile3,map3,debug);
  bool check3 = (map3.size() > 0);
  std::ofstream file;
  file.open(outfile.c_str(), std::ofstream::out);
  unsigned int k(0), good(0);
  for (std::map<std::pair<int,int>,int>::iterator itr=map1.begin();
       itr != map1.end(); ++itr, ++k) {
    std::pair<int,int> key = itr->first;
    bool ok = (map2.find(key) != map2.end());
    if (ok && check3) ok = (map3.find(key) != map3.end());
    if (debug && k%100 == 0) 
      std::cout << "[" << k << "] Run " << key.first << " Event " << key.second
		<< " Flag " << ok << std::endl;
    if (ok) {
      ++good;
      file << key.first << "   " << key.second << std::endl;
    }
  }
  file.close();
  std::cout << "Total # of common events " << good << " written to o/p file "
	    << outfile << std::endl;
}

void readRecordEvents(std::string fname, std::vector<recordEvent>& records,
		      bool debug) {
  records.clear();
  ifstream infile (fname.c_str());
  if (!infile.is_open()) {
    std::cout << "Cannot open " << fname << std::endl;
  } else {
    while (1) {
      unsigned int ser, ent, r, ev;
      infile >> ser >> ent >> r >> ev;
      if (!infile.good()) break;
      recordEvent rec(ser,ent,r,ev);
      records.push_back(rec);
    }
    infile.close();
  }
  std::cout << "Reads " << records.size() << " records from " << fname
	    << std::endl;
  if (debug) {
    for (unsigned int k=0; k<records.size(); ++k) {
      if (k%100 == 0) 
	std::cout << "[" << records[k].serial_ << ":" << records[k].entry_ 
		  << "] " << records[k].run_ << ":" << records[k].event_ 
		  << std::endl;
    }
  }
}

void sortEvent(std::vector<recordEvent>& records, bool debug) {
  // Use std::sort
  std::sort(records.begin(), records.end(), recordEventLess());
  if (debug) {
    for (unsigned int k=0; k<records.size(); ++k) {
      std::cout << "[" << k << ":" << records[k].serial_ << ":" 
		<< records[k].entry_ << "] " << records[k].run_ << ":" 
		<< records[k].event_ << std::endl;
    }
  }
}


void duplicateEvent(std::string fname, std::vector<recordEvent>& records,
		    bool debug) {
  std::ofstream file;
  file.open(fname.c_str(), std::ofstream::out);
  std::cout << "List of entry names of duplicate events" << std::endl;
  int duplicate(0);
  for (unsigned int k=1; k<records.size(); ++k) {
    if ((records[k].run_ == records[k-1].run_) &&
	(records[k].event_ == records[k-1].event_)) {
      // This is a duplicate event - reject the one with larger serial #
      if (records[k].entry_ < records[k-1].entry_) {
	recordEvent swap = records[k-1];
	records[k-1]= records[k];
	records[k]  = swap;
      }
      if (debug) {
	std::cout << "Serial " << records[k-1].serial_ << ":"  
		  << records[k].serial_ << " Entry "  
		  << records[k-1].entry_ << ":" << records[k].entry_ << " Run "
		  << records[k-1].run_ << ":"  << records[k].run_ << " Event "
		  << records[k-1].event_ << " " << records[k].event_
		  << std::endl;
      }
      file << records[k].entry_ << std::endl;
      duplicate++;
    }
  }
  file.close();
  std::cout << "Total # of duplcate events " << duplicate << std::endl;
}

void findDuplicateEvent(std::string infile, std::string outfile,
			bool debug=false) {

  std::vector<recordEvent> records;
  readRecordEvents(infile, records, debug);
  sortEvent(records,debug);
  duplicateEvent(outfile, records, debug);
}
