//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibMonitor.C+g
//  CalibMonitor c1(fname, dirname, dupFileName, outFileName, prefix, 
//                  corrFileName, flag, numb, dataMC, useGen, scale,
//                  etalo, etahi, runlo, runhi, phimin, phimax, zside);
//  c1.Loop();
//  c1.SavePlot(histFileName,append,all);
//
//        This will prepare a set of histograms which can be used for a
//        quick fit and display using the methods in CalibFitPlots.C
//
//  GetEntries g1(fname, dirname, bool ifOld);
//  g1.Loop();
//
//         This looks into the tree *EventInfo* and can provide a set
//         of histograms with event statistics
//
//   where:
// 
//   fname   (std::string)     = file name of the input ROOT tree
//   dirname (std::string)     = name of the directory where Tree resides
//                               (use "HcalIsoTrkAnalyzer")
//   dupFileName (std::string) = name of the file containing list of entries 
//                               of duplicate events
//   outFileName (std::string) = name of a text file to be created (under
//                               control of value of flag) with information
//                               about events
//   prefix (std::string)      = String to be added to the name of histogram
//                               (usually a 4 character string; default="")
//   corrFileName (std::string)= name of the text file having the correction
//                               factors to be used (default="", no corr.)
//   flag (int)                = 4 digit integer (thdo) with specific control
//                               information (t=0/1 for doing or not the PU
//                               correction; h = 0/1/2 for not creating/
//                               creating in recreate mode/creating in append 
//                               mode the output text file; d = 0/1/2/3 
//                               produces 3 standard (0,1,2) or extended (3) 
//                               set of histograms; o = 0/1/2 for tight/loose/
//                               flexible selection). Default = 0
//   numb   (int)              = number of eta bins (42 for -21:21)
//   dataMC (bool)             = true/false for data/MC (default true)
//   useGen (bool)             = true/false to use generator level momentum
//                               or reconstruction level momentum (def false)
//   scale (double)            = energy scale if correction factor to be used
//                               (default = 1.0)
//   etalo/etahi (int,int)     = |eta| ranges (0:30)
//   runlo  (int)              = lower value of run number (def -1)
//   runhi  (int)              = higher value of run number (def 9999999)
//   phimin          (int)     = minimum iphi value (1)
//   phimax          (int)     = maximum iphi value (72)
//   zside           (int)     = the side of the detector if phimin and phimax
//                               differ from 1-72 (1)
//
//   histFileName (std::string)= name of the file containing saved histograms
//   append (bool)             = true/false if the hitogram file to be opened
//                               in append/output mode
//   all (bool)                = true/false if all histograms to be saved or
//                               not (def false)
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
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class CalibMonitor {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

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

  struct record {
    record() {
      serial_ = entry_ = run_ = event_ = ieta_ = p_ = 0;
    };
    record(int ser, Long64_t ent, int r, int ev, int ie, double p) :
      serial_(ser), entry_(ent), run_(r), event_(ev), ieta_(ie), p_(p) {};
    int      serial_;
    Long64_t entry_;
    int      run_, event_, ieta_;
    double   p_;
  };

  CalibMonitor(std::string fname, std::string dirname, 
	       std::string dupFileName, std::string outTxtFileName, 
	       std::string prefix="", std::string corrFileName="",
	       int flag=0, int numb=42, bool datMC=true, bool useGen=false,
	       double scale=1.0, int etalo=0, int etahi=30, int runlo=-1, 
	       int runhi=99999999, int phimin=1, int phimax=72, int zside=1);
  virtual ~CalibMonitor();
  virtual Int_t              Cut(Long64_t entry);
  virtual Int_t              GetEntry(Long64_t entry);
  virtual Long64_t           LoadTree(Long64_t entry);
  virtual void               Init(TTree *tree, std::string& dupFileName);
  virtual void               Loop();
  virtual Bool_t             Notify();
  virtual void               Show(Long64_t entry = -1);
  bool                       GoodTrack (double &eHcal, double &cut, bool debug);
  bool                       SelectPhi(bool debug);
  void                       PlotHist(int type, int num, bool save=false);
  template<class Hist> void  DrawHist(Hist*, TCanvas*);
  void                       SavePlot(std::string theName, bool append, bool all=false);
  bool                       ReadCorrFactor(std::string &fName);
  std::vector<std::string>   SplitString (const std::string& fLine);
private:

  static const unsigned int npbin=5, kp50=2;
  std::string               fname_, dirnm_, prefix_, outTxtFileName_;
  int                       flag_, numb_, flexibleSelect_;
  bool                      dataMC_, useGen_, corrPU_, corrE_;
  int                       plotType_, etalo_, etahi_, runlo_, runhi_;
  int                       phimin_, phimax_, zside_;
  double                    scale_, log2by18_;
  std::vector<Long64_t>     entries_;
  std::vector<double>       etas_, ps_, dl1_;
  std::vector<int>          nvx_, ietas_;
  TH1D                     *h_p[5], *h_eta[5];
  std::vector<TH1D*>        h_eta0, h_eta1, h_eta2, h_eta3, h_eta4;
  std::vector<TH1D*>        h_dL1,  h_vtx, h_etaF[npbin], h_etaB[npbin];
  std::vector<TProfile*>    h_etaX[npbin];
  std::vector<TH1D*>        h_etaR[npbin], h_nvxR[npbin], h_dL1R[npbin];
  std::map<std::pair<int,int>,double> cfactors_;
};

CalibMonitor::CalibMonitor(std::string fname, std::string dirnm, 
			   std::string dupFileName, std::string outTxtFileName,
			   std::string prefix, std::string corrFileName,
			   int flag, int numb, bool dataMC, bool useGen, 
			   double scale, int etalo, int etahi, int runlo,
			   int runhi, int phimin, int phimax,
			   int zside) : fname_(fname), dirnm_(dirnm),
					prefix_(prefix), 
					outTxtFileName_(outTxtFileName),
					flag_(flag), numb_(numb),
					dataMC_(dataMC), useGen_(useGen), 
					etalo_(etalo), etahi_(etahi), 
					runlo_(runlo), runhi_(runhi),
					phimin_(phimin), phimax_(phimax),
					zside_(zside), scale_(scale) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree

  plotType_        = ((flag_/10)%10);
  if (plotType_ < 0 || plotType_ > 3) plotType_ = 3;
  flexibleSelect_  = (((flag_/1) %10));
  corrPU_          = (((flag_/1000) %10) > 0);
  log2by18_        = std::log(2.5)/18.0;
  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir 
	    << " flags " << flexibleSelect_ << "|" << plotType_ << "|"
	    << corrPU_ << " cons " << log2by18_ << " eta range " << etalo_ 
	    << ":" << etahi_ << " run range " << runlo_ << ":" << runhi_ 
	    << std::endl;
  TTree      *tree = (TTree*)dir->Get("CalibTree");
  std::cout << "CalibMonitor:Tree " << tree << std::endl;
  Init(tree,dupFileName);
  corrE_ = ReadCorrFactor(corrFileName);
  std::cout << "Reads correction factors from " << corrFileName << " with flag "
	    << corrE_ << std::endl;
}

CalibMonitor::~CalibMonitor() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t CalibMonitor::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibMonitor::LoadTree(Long64_t entry) {
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

void CalibMonitor::Init(TTree *tree, std::string& dupFileName) {
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
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
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

  ifstream infil1(dupFileName.c_str());
  if (!infil1.is_open()) {
    std::cout << "Cannot open " << dupFileName << std::endl;
  } else {
    while (1) {
      Long64_t jentry;
      infil1 >> jentry;
      if (!infil1.good()) break;
      entries_.push_back(jentry);
    }
    infil1.close();
    std::cout << "Reads a list of " << entries_.size() << " events from " 
	      << dupFileName << std::endl;
  }

  double xbins[99];
  int    nbins(-1);
  if (plotType_ == 0) {
    double xbin[9] = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
    for (int i=0; i<9; ++i) {etas_.push_back(xbin[i]); xbins[i] = xbin[i];}
    nbins = 8;
  } else if (plotType_ == 1) {
    double xbin[11] = {-25.0, -20.0, -15.0, -10.0, -5.0, 0.0,
		       5.0, 10.0, 15.0, 20.0, 25.0};
    for (int i=0; i<11; ++i) {etas_.push_back(xbin[i]); xbins[i] = xbin[i];}
    nbins = 10;
  } else if (plotType_ == 2) {
    double xbin[23] = {-23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0,
		       -7.0, -5.0, -3.0, 0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0,
		       15.0, 17.0, 19.0, 21.0, 23.0};
    for (int i=0; i<23; ++i) {etas_.push_back(xbin[i]); xbins[i] = xbin[i];}
    nbins = 22;
  } else {
    double      xbina[99];
    int         neta = numb_/2;
    for (int k=0; k<neta; ++k) {
      xbina[k]       = (k-neta)-0.5;
      xbina[numb_-k] = (neta-k) + 0.5;
    }
    xbina[neta] = 0;
    for (int i=0; i<numb_+1; ++i) {
      etas_.push_back(xbina[i]);
      xbins[i] = xbina[i];
      ++nbins;
    }
  }
  int ipbin[npbin] = {20, 30, 40, 60, 100};
  for (unsigned int i=0; i<npbin; ++i) ps_.push_back((double)(ipbin[i]));
  int npvtx[6]  = {0, 7,10, 13, 16,100};
  for (int i=0; i<6; ++i)  nvx_.push_back(npvtx[i]);
  double dl1s[9]= {0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  int ietas[4] = {0, 13, 18, 25};
  for (int i=0; i<4; ++i)  ietas_.push_back(ietas[i]);

  char        name[20], title[200];
  std::string titl[5] = {"All tracks", "Good quality tracks", "Selected tracks",
			 "Tracks with charge isolation", "Tracks MIP in ECAL"};
  for (int i=0; i<9; ++i)  dl1_.push_back(dl1s[i]);
  if (plotType_ <= 1) {
    std::cout << "Book Histos for Standard\n";
    for (int k=0; k<5; ++k) {
      sprintf (name, "%sp%d", prefix_.c_str(), k);
      sprintf (title,"%s", titl[k].c_str());
      h_p[k] = new TH1D(name, title, 100, 10.0, 110.0);
      sprintf (name, "%seta%d", prefix_.c_str(), k);
      sprintf (title,"%s", titl[k].c_str());
      h_eta[k] = new TH1D(name, title, 60, -30.0, 30.0);
    }
    unsigned int kp = (ps_.size()-1);
    for (unsigned int k=0; k<kp; ++k) {
      sprintf (name, "%seta0%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[0].c_str(),ipbin[k],ipbin[k+1]);
      h_eta0.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta1%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[1].c_str(),ipbin[k],ipbin[k+1]);
      h_eta1.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta2%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[2].c_str(),ipbin[k],ipbin[k+1]);
      h_eta2.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta3%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[3].c_str(),ipbin[k],ipbin[k+1]);
      h_eta3.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta4%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
      h_eta4.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%sdl1%d", prefix_.c_str(), k);
      sprintf (title,"Distance from L1 (p = %d:%d GeV)",ipbin[k],ipbin[k+1]);
      h_dL1.push_back(new TH1D(name, title, 160, 0.0, 8.0));
      sprintf (name, "%svtx%d", prefix_.c_str(), k);
      sprintf (title,"N_{Vertex} (p = %d:%d GeV)",ipbin[k],ipbin[k+1]);
      h_vtx.push_back(new TH1D(name, title, 100, 0.0, 100.0));
      for (unsigned int i=0; i<nvx_.size(); ++i) {
	sprintf (name, "%setaX%d%d", prefix_.c_str(), k, i);
	if (i == 0) {
	  sprintf (title,"%s (p = %d:%d GeV all vertices)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"%s (p = %d:%d GeV # Vtx %d:%d)",titl[4].c_str(),ipbin[k],ipbin[k+1],nvx_[i-1],nvx_[i]);
	}
	h_etaX[k].push_back(new TProfile(name, title, nbins, xbins));
	unsigned int kk = h_etaX[k].size()-1;
	h_etaX[k][kk]->Sumw2();
	sprintf (name, "%snvxR%d%d", prefix_.c_str(), k, i);
	if (i == 0) {
	  sprintf (title,"E/p for %s (p = %d:%d GeV all vertices)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"E/p for %s (p = %d:%d GeV # Vtx %d:%d)",titl[4].c_str(),ipbin[k],ipbin[k+1],nvx_[i-1],nvx_[i]);
	}
	h_nvxR[k].push_back(new TH1D(name,title,100,0.,5.));
	kk = h_nvxR[k].size()-1;
	h_nvxR[k][kk]->Sumw2();
      }
      for (unsigned int j=0; j<etas_.size(); ++j) {
	sprintf (name, "%sratio%d%d", prefix_.c_str(), k, j);
	if (j == 0) {
	  sprintf (title,"E/p for %s (p = %d:%d GeV)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"E/p for %s (p = %d:%d GeV #eta %4.1f:%4.1f)",titl[4].c_str(),ipbin[k],ipbin[k+1],etas_[j-1],etas_[j]);
	}
	h_etaF[k].push_back(new TH1D(name,title,100,0.,5.));
	unsigned int kk = h_etaF[k].size()-1;
	h_etaF[k][kk]->Sumw2();
	sprintf (name, "%setaR%d%d", prefix_.c_str(), k, j);
	h_etaR[k].push_back(new TH1D(name,title,100,0.,5.));
	kk = h_etaR[k].size()-1;
	h_etaR[k][kk]->Sumw2();
      }
      for (unsigned int j=1; j<ietas_.size(); ++j) {
	sprintf (name, "%setaB%d%d", prefix_.c_str(), k, j);
	sprintf (title,"E/p for %s (p = %d:%d GeV |#eta| %d:%d)",titl[4].c_str(),ipbin[k],ipbin[k+1],ietas_[j-1],ietas_[j]);
	h_etaB[k].push_back(new TH1D(name,title,100,0.,5.));
	unsigned int kk = h_etaB[k].size()-1;
	h_etaB[k][kk]->Sumw2();
      }
      for (unsigned int j=0; j<dl1_.size(); ++j) {
	sprintf (name, "%sdl1R%d%d", prefix_.c_str(), k, j);
	if (j == 0) {
	  sprintf (title,"E/p for %s (p = %d:%d GeV All d_{L1})",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"E/p for %s (p = %d:%d GeV d_{L1} %4.2f:%4.2f)",titl[4].c_str(),ipbin[k],ipbin[k+1],dl1_[j-1],dl1_[j]);
	}
	h_dL1R[k].push_back(new TH1D(name,title,100,0.,5.));
	unsigned int kk = h_dL1R[k].size()-1;
	h_dL1R[k][kk]->Sumw2();
      }
    }
    for (unsigned int i=0; i<nvx_.size(); ++i) {
      sprintf (name, "%setaX%d%d", prefix_.c_str(), kp, i);
      if (i == 0) {
	sprintf (title,"%s (All Momentum all vertices)",titl[4].c_str());
      } else {
	sprintf (title,"%s (All Momentum # Vtx %d:%d)",titl[4].c_str(),nvx_[i-1],nvx_[i]);
      }
      h_etaX[npbin-1].push_back(new TProfile(name, title, nbins, xbins));
      unsigned int kk = h_etaX[npbin-1].size()-1;
      h_etaX[npbin-1][kk]->Sumw2();
      sprintf (name, "%snvxR%d%d", prefix_.c_str(), kp, i);
      if (i == 0) {
	sprintf (title,"E/p for %s (All Momentum all vertices)",titl[4].c_str());
      } else {
	sprintf (title,"E/p for %s (All Momentum # Vtx %d:%d)",titl[4].c_str(),nvx_[i-1],nvx_[i]);
      }
      h_nvxR[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      kk = h_nvxR[npbin-1].size()-1;
      h_nvxR[npbin-1][kk]->Sumw2();
    }
    for (unsigned int j=0; j<etas_.size(); ++j) {
      sprintf (name, "%sratio%d%d", prefix_.c_str(), kp, j);
      if (j == 0) {
	sprintf (title,"E/p for %s (All momentum)",titl[4].c_str());
      } else {
	sprintf (title,"E/p for %s (All momentum #eta %4.1f:%4.1f)",titl[4].c_str(),etas_[j-1],etas_[j]);
      }
      h_etaF[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      unsigned int kk = h_etaF[npbin-1].size()-1;
      h_etaF[npbin-1][kk]->Sumw2();
      sprintf (name, "%setaR%d%d", prefix_.c_str(), kp, j);
      h_etaR[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      kk = h_etaR[npbin-1].size()-1;
      h_etaR[npbin-1][kk]->Sumw2();
    }
    for (unsigned int j=1; j<ietas_.size(); ++j) {
      sprintf (name, "%setaB%d%d", prefix_.c_str(), kp, j);
      sprintf (title,"E/p for %s (All momentum |#eta| %d:%d)",titl[4].c_str(),ietas_[j-1],ietas_[j]);
      h_etaB[npbin-1].push_back(new TH1D(name,title,100,0.,5.));
      unsigned int kk = h_etaB[npbin-1].size()-1;
      h_etaB[npbin-1][kk]->Sumw2();
    }
    for (unsigned int j=0; j<dl1_.size(); ++j) {
      sprintf (name, "%sdl1R%d%d", prefix_.c_str(), kp, j);
      if (j == 0) {
	sprintf (title,"E/p for %s (All momentum)",titl[4].c_str());
      } else {
	sprintf (title,"E/p for %s (All momentum d_{L1} %4.2f:%4.2f)",titl[4].c_str(),dl1_[j-1],dl1_[j]);
      }
      h_dL1R[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      unsigned int kk = h_dL1R[npbin-1].size()-1;
      h_dL1R[npbin-1][kk]->Sumw2();
    }
  } else {
    std::cout << "Book Histos for Non-Standard " << etas_.size() << ":" << kp50 << "\n";
    for (unsigned int j=0; j<etas_.size(); ++j) {
      sprintf (name, "%sratio%d%d", prefix_.c_str(), kp50, j);
      if (j == 0) {
	sprintf (title,"E/p for %s (p = %d:%d GeV)",titl[4].c_str(),ipbin[kp50],ipbin[kp50+1]);
      } else {
	sprintf (title,"E/p for %s (p = %d:%d GeV #eta %4.1f:%4.1f)",titl[4].c_str(),ipbin[kp50],ipbin[kp50+1],etas_[j-1],etas_[j]);
      }
      h_etaF[kp50].push_back(new TH1D(name,title,100,0.,5.));
      unsigned int kk = h_etaF[kp50].size()-1;
      h_etaF[kp50][kk]->Sumw2();
    }
    for (unsigned int j=1; j<ietas_.size(); ++j) {
      sprintf (name, "%setaB%d%d", prefix_.c_str(), kp50, j);
      sprintf (title,"E/p for %s (p = %d:%d GeV |#eta| %d:%d)",titl[4].c_str(),ipbin[kp50],ipbin[kp50+1],ietas_[j-1],ietas_[j]);
      h_etaB[kp50].push_back(new TH1D(name,title,100,0.,5.));
      unsigned int kk = h_etaB[kp50].size()-1;
      h_etaB[kp50][kk]->Sumw2();
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
  if (!fChain) return;
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
  if (fChain == 0) return;
  const bool debug(false);

  std::ofstream fileout;
  if (((flag_/100)%10)>0) {
    if (((flag_/100)%10)==2) {
      fileout.open(outTxtFileName_.c_str(), std::ofstream::out);
      std::cout << "Opens " << outTxtFileName_ << " in output mode" <<std::endl;
    } else {
      fileout.open(outTxtFileName_.c_str(), std::ofstream::app);
      std::cout << "Opens " << outTxtFileName_ << " in append mode" <<std::endl;
    }
    fileout << "Input file: " << fname_ << " Directory: " << dirnm_ 
	    << " Prefix: " << prefix_ << "\n";
  }

  // Find list of duplicate events  
  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << std::endl;
  Long64_t nbytes(0), nb(0);
  unsigned int  duplicate(0), good(0), kount(0);
  unsigned int  kp1 = ps_.size() - 1;
  unsigned int  kv1 = 0;
  std::vector<int> kounts(kp1,0);
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    bool select = (std::find(entries_.begin(),entries_.end(),jentry) == entries_.end());
    if (!select) {
      ++duplicate;
      if (debug) std::cout << "Duplicate event " << t_Run << " " << t_Event 
			   << " " << t_p << std::endl;
      continue;
    }
    select = ((t_Run >= runlo_) && (t_Run <= runhi_) && 
	      (fabs(t_ieta) >= etalo_) && (fabs(t_ieta) <= etahi_));
    if (!select) {
      if (debug) 
	std::cout << "Run # " << t_Run << " out of range of " << runlo_ << ":" 
		  << runhi_ << " or " << t_ieta << " out of range of " << etalo_
		  << ":" << etahi_ << std::endl;
      continue;
    }

    // if (Cut(ientry) < 0) continue;
    int kp(-1), jp(-1), jp1(-1);
    double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
    for (unsigned int k=1; k<ps_.size(); ++k ) {
      if (pmom >= ps_[k-1] && pmom < ps_[k]) {
	kp = k - 1; break;
      }
    }
    unsigned int kv  = nvx_.size() - 1;
    for (unsigned int k=1; k<nvx_.size(); ++k ) {
      if (t_goodPV >= nvx_[k-1] && t_goodPV < nvx_[k]) {
	kv = k; break;
      }
    }
    unsigned int kd1 =  0;
    unsigned int kd  = dl1_.size() - 1;
    for (unsigned int k=1; k<dl1_.size(); ++k ) {
      if (t_mindR1 >= dl1_[k-1] && t_mindR1 < dl1_[k]) {
	kd = k; break;
      }
    }
    double eta = (t_ieta > 0) ? ((double)(t_ieta)-0.001) : ((double)(t_ieta)+0.001);
    for (unsigned int j=1; j<etas_.size(); ++j) {
      if (eta > etas_[j-1] && eta < etas_[j]) {
	jp = j; break;
      }
    }
    for (unsigned int j=1; j<ietas_.size(); ++j) {
      if (std::abs(t_ieta) > ietas_[j-1] && std::abs(t_ieta) < ietas_[j]) {
	jp1 = j-1; break;
      }
    }
    if (debug) std::cout << "Bin " << kp << ":" << kp1 << ":" << kv << ":" 
			 << kv1 << ":" << kd << ":" << kd1 << ":" << jp << ":"
			 << jp1 << std::endl;
    if (plotType_ <= 1) {
      h_p[0]->Fill(pmom,t_EventWeight);
      h_eta[0]->Fill(t_ieta,t_EventWeight);
      if (kp >= 0) h_eta0[kp]->Fill(t_ieta,t_EventWeight);
    }
    double cut = (pmom > 20) ? ((flexibleSelect_ == 0) ? 2.0 : 10.0) : 0.0;
    double rcut= (pmom > 20) ? 0.25: 0.1;

    // Some Standard plots for control
    if (plotType_ <= 1) {
      if (t_qltyFlag) {
	h_p[1]->Fill(pmom,t_EventWeight);
	h_eta[1]->Fill(t_ieta,t_EventWeight);
	if (kp >= 0) h_eta1[kp]->Fill(t_ieta,t_EventWeight);
	if (t_selectTk) {
	  h_p[2]->Fill(pmom,t_EventWeight);
	  h_eta[2]->Fill(t_ieta,t_EventWeight);
	  if (kp >= 0) h_eta2[kp]->Fill(t_ieta,t_EventWeight);
	  if (t_hmaxNearP < cut) {
	    h_p[3]->Fill(pmom,t_EventWeight);
	    h_eta[3]->Fill(t_ieta,t_EventWeight);
	    if (kp >= 0) h_eta3[kp]->Fill(t_ieta,t_EventWeight);
	    if (t_eMipDR < 1.0) {
	      h_p[4]->Fill(pmom,t_EventWeight);
	      h_eta[4]->Fill(t_ieta,t_EventWeight);
	      if (kp >= 0) {
		h_eta4[kp]->Fill(t_ieta,t_EventWeight);
		h_dL1[kp]->Fill(t_mindR1,t_EventWeight);
		h_vtx[kp]->Fill(t_goodPV,t_EventWeight);
	      }
	    }
	  }
	}
      }
    }

    // Selection of good track and energy measured in Hcal
    double rat(1.0), eHcal(t_eHcal);
    if (corrE_) {
      eHcal = 0;
      for (unsigned int k=0; k<t_HitEnergies->size(); ++k) {
	int depth  = ((*t_DetIds)[k] >> 20) & (0xF);
	int zside  = ((*t_DetIds)[k]&0x80000)?(1):(-1);
	int ieta   = ((*t_DetIds)[k] >> 10) & (0x1FF);
	std::map<std::pair<int,int>,double>::iterator 
	  itr = cfactors_.find(std::pair<int,int>(zside*ieta,depth));
	double cfac = (itr == cfactors_.end()) ? 1.0 : itr->second;
	eHcal += (cfac*((*t_HitEnergies)[k]));
	if (debug) std::cout << zside << ":" << ieta << ":" << depth 
			     << " Corr " << cfac << " " << (*t_HitEnergies)[k] 
			     << " Out " << eHcal << std::endl;
      }
    }
    bool goodTk = GoodTrack(eHcal, cut, debug);
    bool selPhi = SelectPhi(debug);
    if (pmom > 0) rat =  (eHcal/(pmom-t_eMipDR));
    if (debug) 
      std::cout << "Entry " << jentry << " p|eHcal|ratio " << pmom << "|" 
		<< t_eHcal << "|" << eHcal << "|" << rat << "|" << kp << "|" 
		<< kv << "|" << jp << " Cuts " 	<< t_qltyFlag << "|" 
		<< t_selectTk << "|" << (t_hmaxNearP < cut) << "|" 
		<< (t_eMipDR < 1.0) << "|" << goodTk << "|" << (rat > rcut)
		<< " Select Phi " << selPhi << std::endl;
    if (debug)
      std::cout << "D1 : " << kp << ":" << kp1 << ":" << kv << ":" << kv1
		<< ":" << kd << ":" << kd1 << ":" << jp << std::endl;
    if (goodTk && kp >=0 && selPhi) {
      if (rat > rcut) {
	if (plotType_ <= 1) {
	  h_etaX[kp][kv]->Fill(eta,rat,t_EventWeight);
	  h_etaX[kp][kv1]->Fill(eta,rat,t_EventWeight);
	  h_nvxR[kp][kv]->Fill(rat,t_EventWeight);
	  h_nvxR[kp][kv1]->Fill(rat,t_EventWeight);
	  h_dL1R[kp][kd]->Fill(rat,t_EventWeight);
	  h_dL1R[kp][kd1]->Fill(rat,t_EventWeight);
	  if (jp > 0) h_etaR[kp][jp]->Fill(rat,t_EventWeight);
	  h_etaR[kp][0]->Fill(rat,t_EventWeight);
	}
	if ((!dataMC_) || (t_mindR1 > 0.5) || (t_DataType == 1)) {
	  ++kounts[kp];
	  if (plotType_ <= 1) {
	    if (jp > 0) h_etaF[kp][jp]->Fill(rat,t_EventWeight);
	    h_etaF[kp][0]->Fill(rat,t_EventWeight);
	  } else if (kp == (int)(kp50)) {
	    if (debug) std::cout << "kp " << kp << h_etaF[kp].size() << std::endl;
	    if (jp > 0) h_etaF[kp][jp]->Fill(rat,t_EventWeight);
	    h_etaF[kp][0]->Fill(rat,t_EventWeight);
	    if (jp1 >= 0) h_etaB[kp][jp1]->Fill(rat,t_EventWeight);
	  }
	}
	if (pmom > 20.0) {
	  if (plotType_ <= 1) {
	    h_etaX[kp1][kv]->Fill(eta,rat,t_EventWeight);
	    h_etaX[kp1][kv1]->Fill(eta,rat,t_EventWeight);
	    h_nvxR[kp1][kv]->Fill(rat,t_EventWeight);
	    h_nvxR[kp1][kv1]->Fill(rat,t_EventWeight);
	    h_dL1R[kp1][kd]->Fill(rat,t_EventWeight);
	    h_dL1R[kp1][kd1]->Fill(rat,t_EventWeight);
	    if (jp > 0) h_etaR[kp1][jp]->Fill(rat,t_EventWeight);
	    h_etaR[kp1][0]->Fill(rat,t_EventWeight);
	    if (jp1 >= 0) h_etaB[kp][jp1]->Fill(rat,t_EventWeight);
	  }
	}
      }
    }
    if (pmom > 20.0) {
      kount++;
      if (((flag_/100)%10) != 0) {
	good++;
	fileout << good << " " << jentry << " " << t_Run  << " " 
		<< t_Event << " " << t_ieta << " " << pmom << std::endl;
      }
    }
  }
  if (((flag_/100)%10)>0) {
    fileout.close();
    std::cout << "Writes " << good << " events in the file " << outTxtFileName_
	      << std::endl;
  }
  std::cout << "Finds " << duplicate << " Duplicate events out of " << kount
	    << " evnts in this file with p>20 Gev" << std::endl;
  std::cout << "Number of selected events:" << std::endl;
  for (unsigned int k=1; k<ps_.size(); ++k)
    if (ps_[k] > 21)  std::cout << ps_[k-1] <<":"<< ps_[k] << "     " 
				<< kounts[k-1] << std::endl;
}

bool CalibMonitor::GoodTrack(double& eHcal, double &cuti, bool debug) {

  bool select(true);
  double pmom = (useGen_ && (t_gentrackP>0)) ? t_gentrackP : t_p;
  double cut(cuti);
  if (debug) std::cout << "GoodTrack input " << eHcal << ":" << cut;
  if (flexibleSelect_ > 1) {
    double eta = (t_ieta > 0) ? t_ieta : -t_ieta;
    cut        = 8.0*exp(eta*log2by18_);
  }
  if (corrPU_ && pmom > 0) {
    double ediff = (t_eHcal30-t_eHcal10);
    if (ediff >  0.02*pmom) {
      double a1(-0.35), a2(-0.65);
      if (std::abs(t_ieta) == 25) {
	a2 = -0.30;
      } else if (std::abs(t_ieta) > 25) {
	a1 = -0.45; a2 = -0.10;
      }
      double fac = (1.0+a1*(t_eHcal/pmom)*(ediff/pmom)*(1+a2*(ediff/pmom)));
      eHcal *= fac;
    }
  }
  select = ((t_qltyFlag) && (t_selectTk) && (t_hmaxNearP < cut) &&
	    (t_eMipDR < 1.0));
  if (debug) std::cout << " output " << eHcal << ":" << cut << ":" << select 
		       << std::endl;
  return select;
}

bool CalibMonitor::SelectPhi(bool debug) {

  bool   select(true);
  if (phimin_ > 1 || phimax_ < 72) {
    double eTotal(0), eSelec(0);
    for (unsigned int k=0; k<t_HitEnergies->size(); ++k) {
      int iphi  = ((*t_DetIds)[k]) & (0x3FF);
      int zside = ((*t_DetIds)[k]&0x80000)?(1):(-1);
      eTotal   += ((*t_HitEnergies)[k]);
      if (iphi >= phimin_ && iphi <= phimax_ && zside == zside_)
	eSelec += ((*t_HitEnergies)[k]);
    }
    if (eSelec < 0.9*eTotal) select = false;
    if (debug) std::cout << "Etotal " << eTotal << " and ESelec " << eSelec
			 << " (phi " << phimin_ << ":" << phimax_ << " z "
			 << zside_ << ") Selection " << select << std::endl;
  }
  return select;
}

void CalibMonitor::PlotHist(int itype, int inum, bool save) {
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);     gStyle->SetOptFit(1);
  char name[100];
  int itmin = (itype >=0 && itype < 14) ? itype : 0;
  int itmax = (itype >=0 && itype < 14) ? itype : 13;
  std::string types[14] = {"p","i#eta","i#eta","i#eta","i#eta","i#eta",
			   "i#eta","i#eta","E_{HCAL}/(p-E_{ECAL})",
			   "E_{HCAL}/(p-E_{ECAL})","E_{HCAL}/(p-E_{ECAL})",
			   "E_{HCAL}/(p-E_{ECAL})","dR_{L1}","Vertex"};
  int nmax[14] = {5, 5, npbin-1, npbin-1, npbin-1, npbin-1, npbin-1, 
		  npbin, npbin, npbin, npbin, npbin, npbin-1,npbin-1};
  for (int type=itmin; type<=itmax; ++type) {
    int inmin = (inum >=0 && inum < nmax[type]) ? inum : 0;
    int inmax = (inum >=0 && inum < nmax[type]) ? inum : nmax[type]-1;
    int kmax  = 1;
    if      (type == 8)  kmax = (int)(etas_.size());
    else if (type == 7)  kmax = (int)(nvx_.size());
    else if (type == 9)  kmax = (int)(etas_.size());
    else if (type == 10) kmax = (int)(nvx_.size());
    else if (type == 11) kmax = (int)(dl1_.size());
    for (int num=inmin; num<=inmax; ++num) {
      for (int k=0; k<kmax; ++k) {
	sprintf (name, "c_%d%d%d", type, num, k);
	TCanvas *pad = new TCanvas(name, name, 700, 500);
	pad->SetRightMargin(0.10);
	pad->SetTopMargin(0.10);
	sprintf (name, "%s", types[type].c_str());
	if (type != 7) {
	  TH1D* hist(0);
	  if      (type == 0)  hist = (TH1D*)(h_p[num]->Clone());
	  else if (type == 1)  hist = (TH1D*)(h_eta[num]->Clone());
	  else if (type == 2)  hist = (TH1D*)(h_eta0[num]->Clone());
	  else if (type == 3)  hist = (TH1D*)(h_eta1[num]->Clone());
	  else if (type == 4)  hist = (TH1D*)(h_eta2[num]->Clone());
	  else if (type == 5)  hist = (TH1D*)(h_eta3[num]->Clone());
	  else if (type == 6)  hist = (TH1D*)(h_eta4[num]->Clone());
	  else if (type == 8)  hist = (TH1D*)(h_etaR[num][k]->Clone());
	  else if (type == 9)  hist = (TH1D*)(h_etaF[num][k]->Clone());
	  else if (type == 10) hist = (TH1D*)(h_nvxR[num][k]->Clone());
	  else if (type == 11) hist = (TH1D*)(h_dL1R[num][k]->Clone());
	  else if (type == 12) hist = (TH1D*)(h_dL1[num]->Clone());
	  else                 hist = (TH1D*)(h_vtx[num]->Clone());
	  hist->GetXaxis()->SetTitle(name);
	  hist->GetYaxis()->SetTitle("Tracks");
	  DrawHist(hist,pad);
	  if (save) {
	    sprintf (name, "c_%s%d%d%d.gif", prefix_.c_str(), type,num,k);
	    pad->Print(name);
	  }
	} else {
	  TProfile* hist = (TProfile*)(h_etaX[num][k]->Clone());
	  hist->GetXaxis()->SetTitle(name);
	  hist->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	  hist->GetYaxis()->SetRangeUser(0.4,1.6);
	  hist->Fit("pol0","q");
	  DrawHist(hist,pad);
	  if (save) {
	    sprintf (name, "c_%s%d%d%d.gif", prefix_.c_str(), type,num,k);
	    pad->Print(name);
	  }
	}
      }	
    }
  }
}

bool CalibMonitor::ReadCorrFactor(std::string &fname) {
  bool ok(false);
  if (fname != "") {
    std::ifstream fInput(fname.c_str());
    if (!fInput.good()) {
      std::cout << "Cannot open file " << fname << std::endl;
    } else {
      char buffer [1024];
      unsigned int all(0), good(0);
      while (fInput.getline(buffer, 1024)) {
	++all;
	if (buffer [0] == '#') continue; //ignore comment
	std::vector <std::string> items = SplitString (std::string (buffer));
	if (items.size () != 5) {
	  std::cout << "Ignore  line: " << buffer << std::endl;
	} else {
	  ++good;
	  int   ieta  = std::atoi (items[1].c_str());
	  int   depth = std::atoi (items[2].c_str());
	  float corrf = std::atof (items[3].c_str());
	  cfactors_[std::pair<int,int>(ieta,depth)] = scale_*corrf;
	}
      }
      fInput.close();
      std::cout << "Reads total of " << all << " and " << good 
		<< " good records" << std::endl;
      if (good > 0) ok = true;
    }
  }
  return ok;
}

std::vector<std::string> CalibMonitor::SplitString (const std::string& fLine) {
  std::vector <std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine [i] == ' ' || i == fLine.size ()) {
      if (!empty) {
	std::string item (fLine, start, i-start);
	result.push_back (item);
	empty = true;
      }
      start = i+1;
    } else {
      if (empty) empty = false;
    }
  }
  return result;
}

template<class Hist> void CalibMonitor::DrawHist(Hist* hist, TCanvas* pad) {
  hist->GetYaxis()->SetLabelOffset(0.005);
  hist->GetYaxis()->SetTitleOffset(1.20);
  hist->Draw();
  pad->Update();
  TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
  if (st1 != NULL) {
    st1->SetY1NDC(0.70); st1->SetY2NDC(0.90);
    st1->SetX1NDC(0.55); st1->SetX2NDC(0.90);
  }
  pad->Modified();
  pad->Update();
}

void CalibMonitor::SavePlot(std::string theName, bool append, bool all) {

  TFile* theFile(0);
  if (append) {
    theFile = new TFile(theName.c_str(), "UPDATE");
  } else {
    theFile = new TFile(theName.c_str(), "RECREATE");
  }

  theFile->cd();
  for (unsigned int k=0; k<ps_.size(); ++k) {
    if (plotType_ <= 1) {
      for (unsigned int i=0; i<nvx_.size(); ++i) {
	if (h_etaX[k][i] != 0) {
	  TProfile* hnew = (TProfile*)h_etaX[k][i]->Clone();
	  hnew->Write();
	}
	if (h_nvxR[k].size() > i && h_nvxR[k][i] != 0) {
	  TH1D* hist = (TH1D*)h_nvxR[k][i]->Clone();
	  hist->Write();
	}
      }
    }
    for (unsigned int j=0; j<etas_.size(); ++j) {
      if ((plotType_ <= 1) && (h_etaR[k][j] != 0)) {
	TH1D* hist = (TH1D*)h_etaR[k][j]->Clone();
	hist->Write();
      }
      if (h_etaF[k].size() > j && h_etaF[k][j] != 0) {
	TH1D* hist = (TH1D*)h_etaF[k][j]->Clone();
	hist->Write();
      }
    }
    if (plotType_ <= 1) {
      for (unsigned int j=0; j<dl1_.size(); ++j) {
	if (h_dL1R[k][j] != 0) {
	  TH1D* hist = (TH1D*)h_dL1R[k][j]->Clone();
	  hist->Write();
	}
      }
    }
    for (unsigned int j=0; j<ietas_.size(); ++j) {
      if (h_etaB[k].size() > j && h_etaB[k][j] != 0) {
	TH1D* hist = (TH1D*)h_etaB[k][j]->Clone();
	hist->Write();
      }
    }
    if (all && (plotType_ <= 1) && ((k+1) < ps_.size())) {
      if (h_eta0[k] != 0) {TH1D* h1 = (TH1D*)h_eta0[k]->Clone(); h1->Write();}
      if (h_eta1[k] != 0) {TH1D* h2 = (TH1D*)h_eta1[k]->Clone(); h2->Write();}
      if (h_eta2[k] != 0) {TH1D* h3 = (TH1D*)h_eta2[k]->Clone(); h3->Write();}
      if (h_eta3[k] != 0) {TH1D* h4 = (TH1D*)h_eta3[k]->Clone(); h4->Write();}
      if (h_eta4[k] != 0) {TH1D* h5 = (TH1D*)h_eta4[k]->Clone(); h5->Write();}
      if (h_dL1[k] != 0)  {TH1D* h6 = (TH1D*)h_dL1[k]->Clone();  h6->Write();}
      if (h_vtx[k] != 0)  {TH1D* h7 = (TH1D*)h_vtx[k]->Clone();  h7->Write();}
    }
  }
  std::cout << "All done\n";
  theFile->Close();
}

class GetEntries {
public :
  TTree                     *fChain;   //!pointer to the analyzed TTree/TChain
  Int_t                      fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t                      t_Tracks;
  Int_t                      t_TracksProp;
  Int_t                      t_TracksSaved;
  Int_t                      t_TracksLoose;
  Int_t                      t_TracksTight;
  Bool_t                     t_TrigPass;
  Bool_t                     t_TrigPassSel;
  Bool_t                     t_L1Bit;
  std::vector<int>          *t_ietaAll;
  std::vector<int>          *t_ietaGood;

  // List of branches
  TBranch                   *b_t_Tracks;        //!
  TBranch                   *b_t_TracksProp;    //!
  TBranch                   *b_t_TracksSaved;   //!
  TBranch                   *b_t_TracksLoose;   //!
  TBranch                   *b_t_TracksTight;   //!
  TBranch                   *b_t_TrigPass;      //!
  TBranch                   *b_t_TrigPassSel;   //!
  TBranch                   *b_t_L1Bit;         //!
  TBranch                   *b_t_ietaAll;       //!
  TBranch                   *b_t_ietaGood;      //!

  GetEntries(std::string fname, std::string dirname, bool ifOld=false);
  virtual ~GetEntries();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

private:
  bool             ifOld_;
  TH1I            *h_tk[3], *h_eta[2];
  TH1D            *h_eff;
};

GetEntries::GetEntries(std::string fname, std::string dirnm, bool ifOld) : ifOld_(ifOld) {

  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir << std::endl;
  TTree      *tree = (TTree*)dir->Get("EventInfo");
  std::cout << "CalibTree " << tree << std::endl;
  Init(tree);
}

GetEntries::~GetEntries() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t GetEntries::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t GetEntries::LoadTree(Long64_t entry) {
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

void GetEntries::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set branch addresses and branch pointers
  // Set object pointer
  t_ietaAll      = 0;
  t_ietaGood     = 0;
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  fChain->SetBranchAddress("t_Tracks",      &t_Tracks,      &b_t_Tracks);
  fChain->SetBranchAddress("t_TracksProp",  &t_TracksProp,  &b_t_TracksProp);
  fChain->SetBranchAddress("t_TracksSaved", &t_TracksSaved, &b_t_TracksSaved);
  fChain->SetBranchAddress("t_TracksLoose", &t_TracksLoose, &b_t_TracksLoose);
  fChain->SetBranchAddress("t_TracksTight", &t_TracksTight, &b_t_TracksTight);
  fChain->SetBranchAddress("t_TrigPass",    &t_TrigPass,    &b_t_TrigPass);
  fChain->SetBranchAddress("t_TrigPassSel", &t_TrigPassSel, &b_t_TrigPassSel);
  fChain->SetBranchAddress("t_L1Bit",       &t_L1Bit,       &b_t_L1Bit);
  if (!ifOld_) {
    fChain->SetBranchAddress("t_ietaAll",     &t_ietaAll,     &b_t_ietaAll);
    fChain->SetBranchAddress("t_ietaGood",    &t_ietaGood,    &b_t_ietaGood);
  }
  Notify();

  h_tk[0] = new TH1I("Track0", "# of tracks produced",      2000, 0, 2000);
  h_tk[1] = new TH1I("Track1", "# of tracks propagated",    2000, 0, 2000);
  h_tk[2] = new TH1I("Track2", "# of tracks saved in tree", 2000, 0, 2000);
  h_eta[0] = new TH1I("Eta0", "i#eta (All Tracks)",           60, -30, 30);
  h_eta[1] = new TH1I("Eta1", "i#eta (Good Tracks)",          60, -30, 30);
  h_eff    = new TH1D("Eta2", "i#eta (Selection Efficiency)", 60, -30, 30);
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
  if (!fChain) return;
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
  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  int      kount(0), selected(0);
  int      l1(0), hlt(0), loose(0), tight(0);
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    h_tk[0]->Fill(t_Tracks);
    h_tk[1]->Fill(t_TracksProp);
    h_tk[2]->Fill(t_TracksSaved);
    if (t_L1Bit) {
      ++l1;
      if (t_TracksLoose > 0) ++loose;
      if (t_TracksTight > 0) ++tight;
      if (t_TrigPass)        ++hlt;
    }
    if (t_TrigPass) { 
      ++kount;
      if (t_TrigPassSel) ++selected;
    }
    if (!ifOld_) {
      for (unsigned int k=0; k<t_ietaAll->size(); ++k)
	h_eta[0]->Fill((*t_ietaAll)[k]);
      for (unsigned int k=0; k<t_ietaGood->size(); ++k)
	h_eta[1]->Fill((*t_ietaGood)[k]);
    }
  }
  double ymaxk(0);
  if (!ifOld_) {
    for (int i=1; i<=h_eff->GetNbinsX(); ++i) {
      double rat(0), drat(0);
      if (h_eta[0]->GetBinContent(i) > ymaxk) ymaxk = h_eta[0]->GetBinContent(i);
      if ((h_eta[1]->GetBinContent(i) > 0)&&(h_eta[0]->GetBinContent(i) > 0)) {
	rat = h_eta[1]->GetBinContent(i)/h_eta[0]->GetBinContent(i);
	drat= rat*std::sqrt(pow((h_eta[1]->GetBinError(i)/h_eta[1]->GetBinContent(i)),2) +
			    pow((h_eta[0]->GetBinError(i)/h_eta[0]->GetBinContent(i)),2));
      }
      h_eff->SetBinContent(i,rat);
      h_eff->SetBinError(i,drat);
    }
  }
  std::cout << "===== " << kount << " events passed trigger of which " 
	    << selected << " events get selected =====\n" << std::endl;
  std::cout << "===== " << l1 << " events passed L1 " << hlt 
	    << " events passed HLT and " << loose << ":" << tight
	    << " events have at least 1 track candidate with loose:tight"
	    << " isolation cut =====\n" << std::endl;
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(1110);       gStyle->SetOptTitle(0);
  int color[3] = {kBlack, kRed, kBlue};
  int lines[3] = {1, 2, 3};
  TCanvas *pad1 = new TCanvas("c_track", "c_track", 500, 500);
  pad1->SetRightMargin(0.10);
  pad1->SetTopMargin(0.10);
  pad1->SetFillColor(kWhite);
  std::string titl1[3] = {"Reconstructed", "Propagated", "Saved"};
  TLegend  *legend1 = new TLegend(0.11, 0.80, 0.50, 0.89);
  legend1->SetFillColor(kWhite);
  double ymax(0), xmax(0);
  for (int k=0; k<3; ++k) {
    int total(0), totaltk(0);
    for (int i=1; i<=h_tk[k]->GetNbinsX(); ++i) {
      if (ymax < h_tk[k]->GetBinContent(i)) ymax = h_tk[k]->GetBinContent(i);
      if (i > 1) total += (int)(h_tk[k]->GetBinContent(i));
      totaltk += (int)(h_tk[k]->GetBinContent(i))*(i-1);
      if (h_tk[k]->GetBinContent(i) > 0) {
	if (xmax < h_tk[k]->GetBinLowEdge(i)+h_tk[k]->GetBinWidth(i))
	  xmax = h_tk[k]->GetBinLowEdge(i)+h_tk[k]->GetBinWidth(i);
      }
    }
    h_tk[k]->SetLineColor(color[k]);
    h_tk[k]->SetMarkerColor(color[k]);
    h_tk[k]->SetLineStyle(lines[k]);
    std::cout << h_tk[k]->GetTitle() << " Entries " << h_tk[k]->GetEntries()
	      << " Events " << total << " Tracks " << totaltk << std::endl;
    legend1->AddEntry(h_tk[k],titl1[k].c_str(),"l");
  }
  int i1 = (int)(0.1*xmax) + 1;
  xmax   = 10.0*i1;
  int i2 = (int)(0.01*ymax) + 1;
  
  ymax   = 100.0*i2;
  for (int k=0; k<3; ++k) {
    h_tk[k]->GetXaxis()->SetRangeUser(0,  xmax);
    h_tk[k]->GetYaxis()->SetRangeUser(0.1,ymax);
    h_tk[k]->GetXaxis()->SetTitle("# Tracks");
    h_tk[k]->GetYaxis()->SetTitle("Events");
    h_tk[k]->GetYaxis()->SetLabelOffset(0.005);
    h_tk[k]->GetYaxis()->SetTitleOffset(1.20);
    if (k == 0) h_tk[k]->Draw("hist");
    else        h_tk[k]->Draw("hist sames");
  }
  pad1->Update();
  pad1->SetLogy();
  ymax = 0.90;
  for (int k=0; k<3; ++k) {
    TPaveStats* st1 = (TPaveStats*)h_tk[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      st1->SetLineColor(color[k]);
      st1->SetTextColor(color[k]);
      st1->SetY1NDC(ymax-0.09); st1->SetY2NDC(ymax);
      st1->SetX1NDC(0.55);      st1->SetX2NDC(0.90);
      ymax -= 0.09;
    }
  }
  pad1->Modified();
  pad1->Update();
  legend1->Draw("same");
  pad1->Update();

  if (!ifOld_) {
    TCanvas *pad2 = new TCanvas("c_ieta", "c_ieta", 500, 500);
    pad2->SetRightMargin(0.10);
    pad2->SetTopMargin(0.10);
    pad2->SetFillColor(kWhite);
    pad2->SetLogy();
    std::string titl2[2] = {"All Tracks", "Selected Tracks"};
    TLegend  *legend2 = new TLegend(0.11, 0.82, 0.50, 0.89);
    legend2->SetFillColor(kWhite);
    i2    = (int)(0.001*ymaxk) + 1;
    ymax  = 1000.0*i2;
    for (int k=0; k<2; ++k) {
      h_eta[k]->GetYaxis()->SetRangeUser(1,ymax);
      h_eta[k]->SetLineColor(color[k]);
      h_eta[k]->SetMarkerColor(color[k]);
      h_eta[k]->SetLineStyle(lines[k]);
      h_eta[k]->GetXaxis()->SetTitle("i#eta");
      h_eta[k]->GetYaxis()->SetTitle("Tracks");
      h_eta[k]->GetYaxis()->SetLabelOffset(0.005);
      h_eta[k]->GetYaxis()->SetTitleOffset(1.20);
      legend2->AddEntry(h_eta[k],titl2[k].c_str(),"l");
      if (k == 0) h_eta[k]->Draw("hist");
      else        h_eta[k]->Draw("hist sames");
    }
    pad2->Update();
    ymax = 0.90;
//  double ymin = 0.10;
    for (int k=0; k<2; ++k) {
      TPaveStats* st1 = (TPaveStats*)h_eta[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetLineColor(color[k]);
	st1->SetTextColor(color[k]);
	st1->SetY1NDC(ymax-0.09); st1->SetY2NDC(ymax);
	st1->SetX1NDC(0.55); st1->SetX2NDC(0.90);
	ymax -= 0.09;
      }
    }
    pad2->Modified();
    pad2->Update();
    legend2->Draw("same");
    pad2->Update();

    TCanvas *pad3 = new TCanvas("c_effi", "c_effi", 500, 500);
    pad3->SetRightMargin(0.10);
    pad3->SetTopMargin(0.10);
    pad3->SetFillColor(kWhite);
    pad3->SetLogy();
    h_eff->SetStats(0);
    h_eff->SetMarkerStyle(20);
    h_eff->GetXaxis()->SetTitle("i#eta");
    h_eff->GetYaxis()->SetTitle("Efficiency");
    h_eff->GetYaxis()->SetLabelOffset(0.005);
    h_eff->GetYaxis()->SetTitleOffset(1.20);
    h_eff->Draw();
    pad3->Modified();
    pad3->Update();
  }
}
