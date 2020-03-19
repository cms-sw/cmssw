////////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibTree.C+g
//  Run(inFileName, dirName, treeName, outFileName, corrFileName, dupFileName,
//      rcorFileName, useweight, useMean, nMin, inverse, ratMin, ratMax, 
//      ietaMax, ietaTrack, sysmode, puCorr, applyL1Cut, l1Cut, truncateFlag,
//      maxIter, corForm, useGen, runlo, runhi, phimin, phimax, zside, nvxlo,
//      nvxhi, rbx, exclude, higheta, fraction, writeDebugHisto, debug);
//
//  where:
//
//  inFileName  (const char*) = file name of the input ROOT tree or name of
//                              the file containing a list of file names of
//                              input ROOT trees ("Silver.root")
//  dirName     (std::string) = name of the directory where the Tree resides
//                              ("HcalIsoTrkAnalyzer")
//  treeName    (std::string) = name of the Tree ("CalibTree")
//  outFileName (std::string) = name of the output ROOT file
//                              ("Silver_out.root")
//  corrFileName(std::string) = name of the output text file with correction
//                              factors ("Silver_corr.txt")
//  dupFileName (std::string) = name of the file containing list of sequence
//                              numbers of duplicate entry ("events_DXS2.txt")
//  rcorFileName (std::string)= name of the text file having the correction
//                              factors as a function of run numbers or depth
//                              to be used for raddam/depth dependent 
//                              correction  (default="", no corr.)
//  useweight       (bool)    = Flag to use event weight (true)
//  useMean         (bool)    = Flag to use Mean of Most probable value
//                              (false -- use MPV)
//  nMin            (int)     = Minmum entries for a given cell which will be
//                              used in evaluating convergence criterion (0)
//  inverse         (bool)    = Use the ratio E/p or p/E in determining the
//                              coefficients (true -- use p/E)
//  ratMin          (double)  = Lower  cut on E/p to select a track (0.25)
//  ratMax          (double)  = Higher cut on E/p to select a track (3.0)
//  ietaMax         (int)     = Maximum ieta value for which correcttion
//                              factor is to be determined (23)
//  ietaTrack       (int)     = Maximum track ieta (at HCAL) to be considered
//                              for these; -1 means no check on track ieta (-1)
//  sysmode         (int)     = systematic error study (-1 if default)
//                              -1 loose, -2 flexible, > 0 for systematic
//  puCorr          (int)     = PU correction to be applied or not: 0 no
//                              correction; < 0 use eDelta; > 0 rho dependent
//                              correction (-1)
//  applyL1Cut      (int)     = Flag to see if closeness to L1 object to be
//                              applied: 0 no check; 1 only to events with
//                              datatype not equal to 1; 2 to all (1)
//  l1Cut           (double)  = Cut value for the closeness parameter (0.5)
//  truncateFlag    (int)     = Flag to treat different depths differently (0)
//                              both depths of ieta 15, 16 of HB as depth 1 (1)
//                              all depths as depth 1 (2), all depths in HE 
//                              with values > 1 as depth 2 (3), all depths in 
//                              HB with values > 1 as depth 2 (4), all depths 
//                              in HB and HE with values > 1 as depth 2 (5)
//                              (Default 0)
//  maxIter         (int)     = number of iterations (30)
//  rcorForm        (int)     = type of rcorFileName: (0) for Raddam correction,
//                              (1) for depth dependent corrections; (2) for
//                              RespCorr corrections; (3) use machine learning
//                              method for pileup correction. (Default 0)
//  useGen          (bool)    = use generator level momentum information (false)
//  runlo           (int)     = lower value of run number to be included (+ve)
//                              or excluded (-ve) (default 0)
//  runhi           (int)     = higher value of run number to be included 
//                              (+ve) or excluded (-ve) (def 9999999)
//  phimin          (int)     = minimum iphi value (1)
//  phimax          (int)     = maximum iphi value (72)
//  zside           (int)     = the side of the detector (0)
//                              (if 0 no selection on zside will be made)
//  nvxlo           (int)     = minimum # of vertex in event to be used (0)
//  nvxhi           (int)     = maximum # of vertex in event to be used (1000)
//  rbx             (int)     = zside*(Subdet*100+RBX #) to be consdered (0)
//  exclude         (bool)    = RBX specified by *rbx* to be exluded or only
//                              considered (false)
//  higheta         (int)     = take correction factors for |ieta| > ietamax
//                              as 1 (0) or of ieta = ietamax with same sign
//                              and depth 1 (1) (default 1)
//  fraction        (double)  = fraction of events to be done (-1)    
//  writeDebugHisto (bool)    = Flag to check writing intermediate histograms
//                              in o/p file (false)
//  debug           (bool)    = To produce more debug printing on screen
//                              (false)
//
//  doIt(inFileName, dupFileName)
//  calls Run 5 times reducing # of events by a factor of 2 in each case
////////////////////////////////////////////////////////////////////////////////

#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TTree.h>
#include <TH1.h>
#include <TGraph.h>
#include <TProfile.h>
#include <algorithm>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#include "CalibCorr.C"

void Run(const char *inFileName="Silver.root",
	 const char *dirName="HcalIsoTrkAnalyzer",
	 const char *treeName="CalibTree",
	 const char *outFileName="Silver_out.root",
	 const char *corrFileName="Silver_corr.txt",
	 const char *dupFileName="events_DXS2.txt", 
	 const char *rcorFileName="",
	 bool useweight=true, bool useMean=false, int nMin=0, bool inverse=true,
	 double ratMin=0.25, double ratMax=3., int ietaMax=23, int ietaTrack=-1,
	 int sysmode=-1, int puCorr=-2, int applyL1Cut=1, double l1Cut=0.5, 
	 int truncateFlag=0, int maxIter=30, int rcorForm=0, bool useGen=false,
	 int runlo=0, int runhi=99999999, int phimin=1, int phimax=72, 
	 int zside=0, int nvxlo=0, int nvxhi=1000, int rbx=0,
	 bool exclude=true, int higheta=1, double fraction=1.0, 
	 bool writeDebugHisto=false, bool debug=false);

// Fixed size dimensions of array or collections stored in the TTree if any.

class CalibTree {

public :

  struct myEntry {
    myEntry (int k=0, double f0=0, double f1=0, 
	     double f2=0) : kount(k), fact0(f0), fact1(f1), fact2(f2) {}
    int    kount;
    double fact0, fact1, fact2;
  };

  struct energyCalor {
    energyCalor (double e1=0, double e2=0, 
		 double e3=0) : Etot(e1), Etot2(e2), ehcal(e3) {}
    double Etot, Etot2, ehcal;
  };

  CalibTree(const char *dupFileName, const char* rcorFileName, int truncateFlag,
	    bool useMean, int runlo, int runhi, int phimin, int phimax,
	    int zside, int nvxlo, int nvxhi, int sysmode, int rbx, int puCorr,
	    int rcorForm, bool useGen, bool exclude, int higheta, 
	    TChain *tree);
  virtual ~CalibTree();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TChain *tree, const char *dupFileName);
  virtual Double_t Loop(int k, TFile *fout, bool useweight, int nMin, 
			bool inverse, double rMin, double rMax, int ietaMax,
			int ietaTrack, int applyL1Cut, double l1Cut, bool last, 
			double fraction, bool writeHisto, bool debug);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  bool             goodTrack();
  void             writeCorrFactor(const char *corrFileName, int ietaMax);
  bool             selectPhi(unsigned int detId);
  std::pair<double,double> fitMean(TH1D*, int);
  void             makeplots(double rmin, double rmax, int ietaMax,
			     bool useWeight, double fraction, bool debug);
  void             fitPol0(TH1D* hist, bool debug);
  void             highEtaFactors(int ietaMax, bool debug);
  energyCalor      energyHcal(double pmom, const Long64_t& entry, bool final);

  TChain                    *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t                      fCurrent;//!current Tree number in a TChain
  TH1D                      *h_pbyE, *h_cvg;
  TProfile                  *h_Ebyp_bfr, *h_Ebyp_aftr;

private:

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

  CalibCorr                                        *cFactor_;
  CalibSelectRBX                                   *cSelect_;
  const int                                         truncateFlag_;
  const bool                                        useMean_;
  int                                               runlo_, runhi_;
  const int                                         phimin_, phimax_;
  const int                                         zside_, nvxlo_, nvxhi_;
  const int                                         sysmode_, rbx_, puCorr_;
  int                                               rcorForm_;
  const bool                                        useGen_, exclude_;
  const int                                         higheta_;
  bool                                              includeRun_;
  double                                            log2by18_, eHcalDelta_;
  std::vector<Long64_t>                             entries;
  std::vector<unsigned int>                         detIds;
  std::map<unsigned int, TH1D*>                     histos;
  std::map<unsigned int, std::pair<double,double> > Cprev;

};


void doIt(const char* infile, const char* dup) {
  char outf1[100], outf2[100];
  double lumt(1.0), fac(0.5);
  for (int k=0; k<5; ++k) {
    sprintf (outf1, "%s_%d.root", infile, k);
    sprintf (outf2, "%s_%d.txt",  infile, k);
    double lumi = (k==0) ? -1 : lumt;
    lumt *= fac;
    Run(infile,"HcalIsoTrkAnalyzer","CalibTree",outf1,outf2,dup,"",true,false,0,
	true,0.25,3.0,25,-1,-1,true,1,0.5,0,30,false,false,-1,99999999,1,72,0,0,
	true,1,lumi,false,false);
  }
}

void Run(const char *inFileName, const char *dirName, const char *treeName, 
	 const char *outFileName, const char *corrFileName,
	 const char *dupFileName, const char* rcorFileName, 
	 bool useweight, bool useMean, int nMin, bool inverse, double ratMin, 
	 double ratMax, int ietaMax, int ietaTrack, int sysmode, int puCorr, 
	 int applyL1Cut, double l1Cut, int truncateFlag, int maxIter, 
	 int rcorForm, bool useGen, int runlo, int runhi, int phimin, 
	 int phimax, int zside, int nvxlo, int nvxhi, int rbx, bool exclude,
	 int higheta, double fraction, bool writeHisto, bool debug) {
 
  char    name[500];
  sprintf (name, "%s/%s", dirName, treeName);
  TChain *chain = new TChain(name);
  std::cout << "Create a chain for " << name << " from " << inFileName
	    << std::endl;
  if (!fillChain(chain,inFileName)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries()
	      << " entries" << std::endl;
    Long64_t nentryTot = chain->GetEntries();
    Long64_t nentries = (fraction > 0.01 && fraction < 0.99) ? 
      (Long64_t)(fraction*nentryTot) : nentryTot;
    static const int maxIterMax = 100;
    if (maxIter > maxIterMax) maxIter = maxIterMax;
    std::cout << "Tree " << name << " " << chain << " in directory " 
	      << dirName << " from file " << inFileName 
	      << " with nentries (tracks): " << nentries << std::endl;
    unsigned int k(0), kmax(maxIter);
    CalibTree t(dupFileName, rcorFileName, truncateFlag, useMean, runlo, runhi, 
		phimin, phimax, zside, nvxlo, nvxhi, sysmode, rbx, puCorr, 
		rcorForm, useGen, exclude, higheta, chain); 
    t.h_pbyE      = new TH1D("pbyE", "pbyE", 100, -1.0, 9.0);
    t.h_Ebyp_bfr  = new TProfile("Ebyp_bfr","Ebyp_bfr",60,-30,30,0,10);
    t.h_Ebyp_aftr = new TProfile("Ebyp_aftr","Ebyp_aftr",60,-30,30,0,10);
    t.h_cvg       = new TH1D("Cvg0", "Convergence", kmax, 0, kmax);
    t.h_cvg->SetMarkerStyle(7);
    t.h_cvg->SetMarkerSize(5.0);
  
    TFile *fout = new TFile(outFileName, "RECREATE");
    std::cout << "Output file: " << outFileName << " opened in recreate mode" 
	      << std::endl;
    fout->cd();

    double cvgs[maxIterMax], itrs[maxIterMax]; 
    for (; k<=kmax; ++k) {
      std::cout << "Calling Loop() "  << k << "th time" << std::endl; 
      double cvg = t.Loop(k, fout, useweight, nMin, inverse, ratMin, ratMax, 
			  ietaMax, ietaTrack, applyL1Cut, l1Cut, k==kmax, 
			  fraction, writeHisto, debug);
      itrs[k] = k;
      cvgs[k] = cvg;
      if (cvg < 0.00001) break;
    }

    t.writeCorrFactor(corrFileName, ietaMax);

    fout->cd();
    TGraph *g_cvg;
    g_cvg = new TGraph(k, itrs, cvgs);
    g_cvg->SetMarkerStyle(7);
    g_cvg->SetMarkerSize(5.0);
    g_cvg->Draw("AP");
    g_cvg->Write("Cvg");
    std::cout << "Finish looping after " << k << " iterations" << std::endl;
    t.makeplots(ratMin, ratMax, ietaMax, useweight, fraction, debug);
    fout->Close();
  }
}

CalibTree::CalibTree(const char *dupFileName, const char* rcorFileName,
		     int flag, bool useMean, int runlo, int runhi, int phimin,
		     int phimax, int zside, int nvxlo, int nvxhi, int mode,
		     int rbx, int pu, int rForm, bool gen, bool excl, int heta,
		     TChain *tree) : fChain(nullptr), cFactor_(nullptr),
				     cSelect_(nullptr), truncateFlag_(flag),
				     useMean_(useMean), runlo_(runlo), 
				     runhi_(runhi), phimin_(phimin),
				     phimax_(phimax), zside_(zside), 
				     nvxlo_(nvxlo), nvxhi_(nvxhi), 
				     sysmode_(mode), rbx_(rbx), puCorr_(pu),
				     rcorForm_(rForm), useGen_(gen), 
				     exclude_(excl), higheta_(heta),
				     includeRun_(true) {
  
  if (runlo_ < 0 || runhi_ < 0) {
    runlo_         = std::abs(runlo_);
    runhi_         = std::abs(runhi_);
    includeRun_    = false;
  }
  log2by18_  = std::log(2.5)/18.0;
  eHcalDelta_= 0;
  std::cout << "Initialize CalibTree with TruncateFlag " << truncateFlag_
	    << " UseMean " << useMean_ << " Run Range " << runlo_ << ":"
	    << runhi_ << " Phi Range " << phimin_ << ":" << phimax_ << ":" 
	    << zside_ << " Vertex Range " << nvxlo_ << ":" << nvxhi_ 
	    << " Mode " << sysmode_ << " PU " << puCorr_ << " rcorFormat "
	    << rcorForm_ << " Gen " << useGen_ << " High Eta " << higheta_ 
	    << std::endl;
  std::cout << "Duplicate events read from " << dupFileName 
	    << " RadDam Corrections read from " << rcorFileName
	    << " Treat RBX " << rbx_ << " with exclusion mode " << exclude_
	    << std::endl;
  Init(tree, dupFileName);
  if (std::string(rcorFileName) != "") {
    cFactor_ = new CalibCorr(rcorFileName,rcorForm_,false);
  } else {
    rcorForm_ = 0;
  }
  if (rbx != 0) cSelect_ = new CalibSelectRBX(rbx);
}

CalibTree::~CalibTree() {
  delete cFactor_;
  delete cSelect_;
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t CalibTree::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibTree::LoadTree(Long64_t entry) {
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

void CalibTree::Init(TChain *tree, const char *dupFileName) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_DetIds       = 0;
  t_HitEnergies  = 0;
  t_trgbits      = 0;
  t_DetIds1      = 0;
  t_DetIds3      = 0;
  t_HitEnergies1 = 0;
  t_HitEnergies3 = 0;
  // Set branch addresses and branch pointers
  fChain = tree;
  if (!tree) return;
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

  if (std::string(dupFileName) != "") {
    ifstream infil1(dupFileName);
    if (!infil1.is_open()) {
      std::cout << "Cannot open " << dupFileName << std::endl;
    } else {
      while (1) {
	Long64_t jentry;
	infil1 >> jentry;
	if (!infil1.good()) break;
	entries.push_back(jentry);
      }
      infil1.close();
      std::cout << "Reads a list of " << entries.size() << " events from " 
		<< dupFileName << std::endl;
    }
  } else {
    std::cout << "No duplicate events in the input file" << std::endl;
  }
}

Bool_t CalibTree::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void CalibTree::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t CalibTree::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

Double_t CalibTree::Loop(int loop, TFile *fout, bool useweight, int nMin,
			 bool inverse, double rmin, double rmax, int ietaMax,
			 int ietaTrack, int applyL1Cut, double l1Cut, bool last,
			 double fraction, bool writeHisto, bool debug) {

  if (fChain == 0) return 0;
  Long64_t nbytes(0), nb(0), kprint(0);
  Long64_t nentryTot = fChain->GetEntriesFast();
  Long64_t nentries = (fraction > 0.01 && fraction < 0.99) ? 
    (Long64_t)(fraction*nentryTot) : nentryTot;
  if (detIds.size() == 0) {
    for (Long64_t jentry=0; jentry<nentries; jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      if (jentry%1000000 == 0) 
	std::cout << "Entry " << jentry << " Run " << t_Run << " Event "
		  << t_Event << std::endl;
      // Find DetIds contributing to the track
      bool selRun = (includeRun_ ? ((t_Run >= runlo_) && (t_Run <= runhi_)) :
		     ((t_Run < runlo_) || (t_Run > runhi_)));
      bool selTrack = ((ietaTrack <= 0) || (abs(t_ieta) <= ietaTrack));
      if (selRun && (t_nVtx >= nvxlo_) && (t_nVtx <= nvxhi_) && selTrack) {
	bool isItRBX(false);
	if (cSelect_ != nullptr) { 
	  bool temp = cSelect_->isItRBX(t_DetIds);
	  if (exclude_) isItRBX = temp;
	  else          isItRBX = !(temp);
	}
	++kprint;
	if (!(isItRBX)) {
	  for (unsigned int idet=0; idet<(*t_DetIds).size(); idet++) { 
	    if (selectPhi((*t_DetIds)[idet])) {
	      unsigned int detid = truncateId((*t_DetIds)[idet],
					      truncateFlag_,debug);
	      if (debug && (kprint<=10)) {
		std::cout << "DetId[" << idet << "] Original " << std::hex 
			  << (*t_DetIds)[idet] << " truncated " << detid 
			  << std::dec;
	      }
	      if (std::find(detIds.begin(),detIds.end(),detid) == detIds.end()){
		detIds.push_back(detid);
		if (debug && (kprint<=10)) std::cout << " new";
	      }
	      if (debug && (kprint<=10)) std::cout << std::endl;
	    }
	  }
	  // Also look at the neighbouring cells if available
	  if (t_DetIds3 != 0) {
	    for (unsigned int idet=0; idet<(*t_DetIds3).size(); idet++) { 
	      if (selectPhi((*t_DetIds3)[idet])) {
		unsigned int detid = truncateId((*t_DetIds3)[idet],
						truncateFlag_,debug);
		if (std::find(detIds.begin(),detIds.end(),detid)==detIds.end()){
		  detIds.push_back(detid);
		}
	      }
	    }
	  }
	}
      }
    }
  }
  if (debug) {
    std::cout << "Total of " << detIds.size() << " detIds and " 
	      << histos.size() << " histos found" << std::endl;
    // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
    for (unsigned int k=0; k<detIds.size(); ++k) {
      int subdet, depth, zside, ieta, iphi;
      unpackDetId(detIds[k], subdet, zside, ieta, iphi, depth);
      std::cout << "DetId[" << k << "] " << subdet << ":" << zside*ieta << ":"
		<< depth << ":" << iphi << "  " << std::hex << detIds[k] 
		<< std::dec << std::endl;
    }
  }
  unsigned int k(0);
  for (std::map<unsigned int, TH1D*>::const_iterator itr = histos.begin();
       itr != histos.end(); ++itr,++k) {
    if (debug) {
      std::cout << "histos[" << k << "] " << std::hex << itr->first 
		<< std::dec << " " << itr->second;
      if (itr->second != 0) std::cout << " " << itr->second->GetTitle();
      std::cout << std::endl;
    }
    if (itr->second != 0) itr->second->Delete();
  }

  for (unsigned int k=0; k<detIds.size(); ++k) {
    char name[20], title[100];
    sprintf (name, "Hist%d_%d", detIds[k], loop);
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detIds[k], subdet, zside, ieta, iphi, depth);
    sprintf (title, "Correction for Subdet %d #eta %d depth %d (Loop %d)", subdet, zside*ieta, depth, loop);
    TH1D* hist = new TH1D(name,title,100, 0.0, 5.0);
    hist->Sumw2();
    if (debug) 
      std::cout << "Book Histo " << k << " " << title << std::endl;
    histos[detIds[k]] = hist;
  }
  std::cout << "Total of " << detIds.size() << " detIds and " << histos.size() 
	    << " found in " << nentries << std::endl;

  nbytes = nb = 0;
  std::map<unsigned int, myEntry > SumW;
  std::map<unsigned int, double  > nTrks;

  int ntkgood(0);
  for (Long64_t jentry=0; jentry<nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)                               break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if (jentry%1000000 == 0) 
      std::cout << "Entry " << jentry << " Run " << t_Run << " Event "
		<< t_Event << std::endl;
    if (std::find(entries.begin(),entries.end(),jentry) !=
	entries.end())                            continue;
    bool selRun = (includeRun_ ? ((t_Run >= runlo_) && (t_Run <= runhi_)) :
		   ((t_Run < runlo_) || (t_Run > runhi_)));
    if (!selRun)                                  continue;
    if ((t_nVtx < nvxlo_) || (t_nVtx > nvxhi_))   continue;
    if (cSelect_ != nullptr) {
      if (exclude_) {
	if (cSelect_->isItRBX(t_DetIds))          continue;
      } else {
	if (!(cSelect_->isItRBX(t_ieta,t_iphi)))  continue;
      }
    }
    bool selTrack = ((ietaTrack <= 0) || (abs(t_ieta) <= ietaTrack));
    if (!selTrack)                                continue;

    if (debug) {
      std::cout << "***Entry (Track) Number : " << ientry << std::endl;
      std::cout << "p/eHCal/eMipDR/nDets : " << t_p << "/" << t_eHcal << "/"
		<< t_eMipDR << "/" << (*t_DetIds).size() << std::endl;
    }
    double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
    if (goodTrack()) {
      ++ntkgood;
      CalibTree::energyCalor en = energyHcal(pmom, jentry, true);
      double evWt = (useweight) ? t_EventWeight : 1.0; 
      if (en.ehcal > 0.001) {
	double pufac = (en.Etot > 0) ? (en.ehcal/en.Etot) : 1.0;
	double ratio = en.ehcal/(pmom-t_eMipDR);
	if (debug) std::cout << " Weights " << evWt << ":" << pufac 
			     << " Energy " << en.Etot2 << ":" << en.Etot << ":" 
			     << pmom << ":" << t_eMipDR << ":" << t_eHcal << ":"
			     << en.ehcal << " ratio " << ratio  << std::endl;
	if (loop==0) {
	  h_pbyE->Fill(ratio, evWt);
	  h_Ebyp_bfr->Fill(t_ieta, ratio, evWt);
	}
	if (last){
	  h_Ebyp_aftr->Fill(t_ieta, ratio, evWt);
	}
	bool l1c(true);
	if (applyL1Cut != 0) l1c = ((t_mindR1 >= l1Cut) || 
				    ((applyL1Cut == 1) && (t_DataType == 1)));
	if ((rmin >=0 && ratio > rmin) && (rmax >= 0 && ratio < rmax) && l1c) {
	  for (unsigned int idet=0; idet<(*t_DetIds).size(); idet++) {
	    if (selectPhi((*t_DetIds)[idet])) {
	      unsigned int id    = (*t_DetIds)[idet];
	      unsigned int detid = truncateId(id,truncateFlag_,false);
	      double hitEn=0.0;
	      if (debug) {
		std::cout << "idet " << idet << " detid/hitenergy : " 
			  << std::hex << (*t_DetIds)[idet] << ":" << detid 
			  << "/" << (*t_HitEnergies)[idet] << std::endl;
	      }
	      if (Cprev.find(detid) != Cprev.end()) 
		hitEn = Cprev[detid].first * (*t_HitEnergies)[idet];
	      else 
		hitEn = (*t_HitEnergies)[idet];
	      if (cFactor_) hitEn *= cFactor_->getCorr(t_Run,id);
	      double Wi  = evWt * hitEn/en.Etot;
	      double Fac = (inverse) ? (en.ehcal/(pmom-t_eMipDR)) : 
		((pmom-t_eMipDR)/en.ehcal);
	      double Fac2= Wi*Fac*Fac;
	      TH1D* hist(0);
	      std::map<unsigned int,TH1D*>::const_iterator itr = histos.find(detid);
	      if (itr != histos.end()) hist = itr->second;
	      if (debug) {
		std::cout << "Det Id " << std::hex << detid << std::dec 
			  << " " << hist << std::endl;
	      }
	      if (hist != 0) hist->Fill(Fac, Wi);//////histola
	      Fac       *= Wi;
	      if (SumW.find(detid) != SumW.end() ) {
		Wi  += SumW[detid].fact0;
		Fac += SumW[detid].fact1;
		Fac2+= SumW[detid].fact2;
		int kount = SumW[detid].kount + 1;
		SumW[detid]   = myEntry(kount,Wi,Fac,Fac2); 
		nTrks[detid] += evWt;
	      } else {
		SumW.insert(std::pair<unsigned int,myEntry>(detid,myEntry(1,Wi,Fac,Fac2)));
		nTrks.insert(std::pair<unsigned int,unsigned int>(detid, evWt));
	      }
	    }
	  }
	}
      }
    }
  }
  if (debug)
    std::cout << "# of Good Tracks " << ntkgood << " out of " << nentries 
	      << std::endl;
  if (loop==0) {
    h_pbyE->Write("h_pbyE");
    h_Ebyp_bfr->Write("h_Ebyp_bfr");
  }
  if (last) {
    h_Ebyp_aftr->Write("h_Ebyp_aftr");
  }

  std::map<unsigned int, std::pair<double,double> > cfactors;
  unsigned int kount(0), kountus(0);
  double       sumfactor(0);
  for (std::map<unsigned int,TH1D*>::const_iterator itr = histos.begin();
       itr != histos.end(); ++itr) {
    if (writeHisto) {
      std::pair<double,double> result_write = fitMean(itr->second, 0);
      (itr->second)->Write();
    }
    // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(itr->first, subdet, zside, ieta, iphi, depth);
    if (debug) {
      std::cout << "DETID :" << subdet << "  IETA :" << ieta 
		<< " HIST ENTRIES :" << (itr->second)->GetEntries()
		<< std::endl;
    }
  }

  if (debug) std::cout << "Histos with " << histos.size() << " entries\n";
  for (std::map<unsigned int,TH1D*>::const_iterator itr = histos.begin();
       itr != histos.end(); ++itr,++kount) {
    std::pair<double,double> result = fitMean(itr->second, 0);
    double factor = (inverse) ? (2.-result.first) : result.first;
    if (debug) {
      int subdet, depth, zside, ieta, iphi;
      unpackDetId(itr->first, subdet, zside, ieta, iphi, depth);
      std::cout << "DetId[" << kount << "] " << subdet << ":" << zside*ieta 
		<< ":" << depth << " Factor " << factor << " +- " 
		<< result.second << std::endl;
    }
    if (!useMean_) {
      cfactors[itr->first] = std::pair<double,double>(factor,result.second);
      if (itr->second->GetEntries() > nMin) {
	kountus++;
	if (factor > 1) sumfactor += (1-1/factor);
	else            sumfactor += (1-factor);
      }
    }
  }
  
  if (debug) std::cout << "SumW with " << SumW.size() << " entries\n";
  std::map<unsigned int, myEntry>::const_iterator SumWItr = SumW.begin();
  for (; SumWItr != SumW.end(); SumWItr++) {
    unsigned int detid = SumWItr->first;
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detid, subdet, zside, ieta, iphi, depth);
    if (debug) {
      std::cout << "Detid|kount|SumWi|SumFac|myId : " << subdet << ":" 
		<< zside*ieta << ":" << depth << " | " 
		<< (SumWItr->second).kount << " | " << (SumWItr->second).fact0
		<< "|" << (SumWItr->second).fact1 << "|" 
		<< (SumWItr->second).fact2 << std::endl;
    }
    double factor = (SumWItr->second).fact1/(SumWItr->second).fact0;
    double dfac1  = ((SumWItr->second).fact2/(SumWItr->second).fact0-factor*factor);
    if (dfac1 < 0) dfac1 = 0;
    double dfac   = sqrt(dfac1/(SumWItr->second).kount);
    if (debug) {
      std::cout << "Factor " << factor << " " << dfac1 << " " << dfac
		<< std::endl;
    }
    if (inverse) factor = 2.-factor;
    if (useMean_) {
      cfactors[detid] = std::pair<double,double>(factor,dfac);
      if ((SumWItr->second).kount > nMin) {
	kountus++;
	if (factor > 1) sumfactor += (1-1/factor);
	else            sumfactor += (1-factor);
      }
    }
  }

  static const int maxch = 500;
  double dets[maxch], cfacs[maxch], wfacs[maxch], myId[maxch], nTrk[maxch];
  std::cout << "cafctors: " << cfactors.size() << ":" << maxch << std::endl;
  kount = 0;
  std::map<unsigned int,std::pair<double,double> >::const_iterator itr=cfactors.begin();
  for (; itr !=cfactors.end(); ++itr,++kount) {
    unsigned int detid = itr->first;
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detid, subdet, zside, ieta, iphi, depth);
    double id  = ieta*zside + 0.25*(depth-1);
    double factor = (itr->second).first;
    double dfac   = (itr->second).second;
    if (ieta > ietaMax) {
      factor = 1;
      dfac   = 0;
    }
    std::pair<double,double> cfac(factor,dfac);
    if (Cprev.find(detid) != Cprev.end()) {
      dfac        /= factor;
      factor      *= Cprev[detid].first;
      dfac        *= factor;
      Cprev[detid] = std::pair<double,double>(factor,dfac);
      cfacs[kount] = factor;
    } else {
      Cprev[detid] = std::pair<double,double>(factor,dfac);
      cfacs[kount] = factor;
    }
    wfacs[kount]= factor;
    dets[kount] = detid;
    myId[kount] = id;
    nTrk[kount] = nTrks[detid];
  }
  if (higheta_ > 0) highEtaFactors(ietaMax, debug);

  std::cout << kountus << " detids out of " << kount << " have tracks > "
	    << nMin << std::endl;

  char fname[50];
  fout->cd();
  TGraph *g_fac1 = new TGraph(kount, dets, cfacs); 
  sprintf (fname, "Cfacs%d", loop);
  g_fac1->SetMarkerStyle(7);
  g_fac1->SetMarkerSize(5.0);
  g_fac1->Draw("AP");
  g_fac1->Write(fname);
  TGraph *g_fac2 = new TGraph(kount, dets, wfacs); 
  sprintf (fname, "Wfacs%d", loop);
  g_fac2->SetMarkerStyle(7);
  g_fac2->SetMarkerSize(5.0);
  g_fac2->Draw("AP");
  g_fac2->Write(fname);
  TGraph *g_fac3 = new TGraph(kount, myId, cfacs); 
  sprintf (fname, "CfacsVsMyId%d", loop);
  g_fac3->SetMarkerStyle(7);
  g_fac3->SetMarkerSize(5.0);
  g_fac3->Draw("AP");
  g_fac3->Write(fname);
  TGraph *g_fac4 = new TGraph(kount, myId, wfacs); 
  sprintf (fname, "WfacsVsMyId%d", loop);
  g_fac4->SetMarkerStyle(7);
  g_fac4->SetMarkerSize(5.0);
  g_fac4->Draw("AP");
  g_fac4->Write(fname);
  TGraph *g_nTrk = new TGraph(kount, myId, nTrk); 
  sprintf (fname, "nTrk");
  if(loop==0){
    g_nTrk->SetMarkerStyle(7);
    g_nTrk->SetMarkerSize(5.0);
    g_nTrk->Draw("AP");
    g_nTrk->Write(fname);
  }
  std::cout << "The new factors are :" << std::endl;
  std::map<unsigned int, std::pair<double,double> >::const_iterator CprevItr = Cprev.begin();
  unsigned int indx(0);
  for (; CprevItr != Cprev.end(); CprevItr++, indx++){
    unsigned int detid = CprevItr->first;
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detid, subdet, zside, ieta, iphi, depth);
    std::cout << "DetId[" << indx << "] " << std::hex << detid << std::dec
	      << "(" << ieta*zside << "," << depth << ") (nTrks:" 
	      << nTrks[detid] << ") : " << CprevItr->second.first << " +- "
	      << CprevItr->second.second << std::endl;
  }
  double mean = (kountus > 0) ? (sumfactor/kountus) : 0;
  std::cout << "Mean deviation " << mean << " from 1 for " << kountus 
	    << " DetIds" << std::endl;
  h_cvg->SetBinContent(loop+1,mean);
  if (last) h_cvg->Write("Cvg0");
  return mean;
}

bool CalibTree::goodTrack() {
  bool   ok(true);
  double cut(2.0);
  double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
  if        (sysmode_ == 1) {
    ok = ((t_qltyFlag) && (t_hmaxNearP < cut) && 
	  (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else if (sysmode_ == 2) {
    ok = ((t_qltyFlag) && (t_qltyPVFlag) && (t_hmaxNearP < cut) && 
	  (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else if (sysmode_ == 3) {
    ok = ((t_selectTk) && (t_hmaxNearP < cut) && 
	  (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else if (sysmode_ == 4) {
    ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < 0.0) && 
	  (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else if (sysmode_ == 5) {
    ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < cut) && 
	  (t_eMipDR < 0.5) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else if (sysmode_ == 6) {
    ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < cut) && 
	  (t_eMipDR < 2.0) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else if (sysmode_ == 7) {
    ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < cut) &&
	  (t_eMipDR < 1.0) && (t_mindR1 > 0.5) && (pmom > 40.0) &&
	  (pmom < 60.0));
  } else {
    if (sysmode_ < 0) {
      double eta = (t_ieta > 0) ? t_ieta : -t_ieta;
      if (sysmode_ == -2) cut = 8.0*exp(eta*log2by18_);
      else                cut = 10.0;
    }
    ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < cut) && 
	  (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (pmom > 40.0) &&
	  (pmom < 60.0));
  }
  return ok;
}

void CalibTree::writeCorrFactor(const char *corrFileName, int ietaMax) {
  ofstream myfile;
  myfile.open(corrFileName);
  if (!myfile.is_open()) {
    std::cout << "** ERROR: Can't open '" << corrFileName << std::endl;
  } else {
    myfile << "#" << std::setprecision(4) << std::setw(10) << "detId" 
	   << std::setw(10) << "ieta" << std::setw(10) << "depth" 
	   << std::setw(15) << "corrFactor" << std::endl;
    std::map<unsigned int, std::pair<double,double> >::const_iterator itr;
    for (itr=Cprev.begin(); itr != Cprev.end(); ++itr) {
      unsigned int detId = itr->first;
      int subdet, depth, zside, ieta, iphi;
      unpackDetId(detId, subdet, zside, ieta, iphi, depth);
      if (ieta <= ietaMax) {
	myfile << std::setw(10) << std::hex << detId << std::setw(10) 
	       << std::dec << zside*ieta << std::setw(10) << depth 
	       << std::setw(10) << itr->second.first << " " << std::setw(10) 
	       << itr->second.second << std::endl;
	std::cout << itr->second.first << ",";
      }
    }
    myfile.close();
    std::cout << std::endl;
  }
}

bool CalibTree::selectPhi(unsigned int detId) {
  bool flag(true);
  // The masks are defined in DataFormats/HcalDetId/interface/HcalDetId.h
  if ((phimin_ > 1) || (phimax_ < 72) || (zside_ != 0)) {
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detId, subdet, zside, ieta, iphi, depth);
    if (phimin_ > 1 || phimax_ < 72) {
      if ((iphi < phimin_) || (iphi > phimax_)) flag = false;
    }
    if (zside_ != 0) {
      if (zside != zside_)                      flag = false;
    }
  }
  return flag;
}

std::pair<double,double> CalibTree::fitMean(TH1D* hist, int mode) {
  std::pair<double,double> results = std::pair<double,double>(1,0);
  if (hist != 0) {
    double mean = hist->GetMean(), rms = hist->GetRMS();
    double LowEdge(0.7), HighEdge(1.3);
    char   option[20];
    if (mode == 1) {
      LowEdge  = mean - 1.5*rms;
      HighEdge = mean + 1.5*rms;
      int nbin = hist->GetNbinsX();
      if (LowEdge < hist->GetBinLowEdge(1)) LowEdge = hist->GetBinLowEdge(1);
      if (HighEdge > hist->GetBinLowEdge(nbin)+hist->GetBinWidth(nbin))
	HighEdge = hist->GetBinLowEdge(nbin)+hist->GetBinWidth(nbin);
    }
    if (hist->GetEntries() > 100) sprintf (option, "+QRLS");
    else                          sprintf (option, "+QRWLS");
    double value(mean);
    double error = rms/sqrt(hist->GetEntries());
    if (hist->GetEntries() > 20) {
      TFitResultPtr Fit = hist->Fit("gaus",option,"",LowEdge,HighEdge);
      value    = Fit->Value(1);
      error    = Fit->FitResult::Error(1); 
      /*
      LowEdge  = value - 1.5*error;
      HighEdge = value + 1.5*error;
      value    = Fit->Value(1);
      error    = Fit->FitResult::Error(1); 
      Fit = hist->Fit("gaus",option,"",LowEdge,HighEdge);
      */
    }
    results =  std::pair<double,double>(value,error);
  }
  return results;
}

void CalibTree::makeplots(double rmin, double rmax, int ietaMax,
			  bool useweight, double fraction, bool debug) {

  if (fChain == 0) return;
  Long64_t nentryTot = fChain->GetEntriesFast();
  Long64_t nentries = (fraction > 0.01 && fraction < 0.99) ? 
    (Long64_t)(fraction*nentryTot) : nentryTot;

  // Book the histograms
  std::map<int,std::pair<TH1D*,TH1D*> > histos;
  for (int ieta=-ietaMax; ieta<=ietaMax; ++ieta) {
    char name[20], title[100];
    sprintf(name,"begin%d",ieta);
    if (ieta==0) sprintf(title,"Ratio at start");
    else         sprintf(title,"Ratio at start for i#eta=%d",ieta);
    TH1D* h1 = new TH1D(name,title,50,rmin,rmax);
    h1->Sumw2();
    sprintf(name,"end%d",ieta);
    if (ieta==0) sprintf(title,"Ratio at the end");
    else         sprintf(title,"Ratio at the end for i#eta=%d",ieta);
    TH1D* h2 = new TH1D(name,title,50,rmin,rmax);
    h2->Sumw2();
    histos[ieta] = std::pair<TH1D*,TH1D*>(h1,h2);
  }
  //Fill the histograms
  Long64_t nbytes(0), nb(0);
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    if (std::find(entries.begin(), entries.end(), jentry) != entries.end()) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if (goodTrack()) {
      double pmom = (useGen_ && (t_gentrackP > 0)) ? t_gentrackP : t_p;
      CalibTree::energyCalor en1 = energyHcal(pmom, jentry, false);
      CalibTree::energyCalor en2 = energyHcal(pmom, jentry, true);
      if ((en1.ehcal > 0.001) && (en2.ehcal > 0.001)) {
	double evWt   = (useweight) ? t_EventWeight : 1.0; 
	double ratioi = en1.ehcal/(pmom-t_eMipDR);
	double ratiof = en2.ehcal/(pmom-t_eMipDR);
	if (t_ieta >= -ietaMax && t_ieta <= ietaMax && t_ieta != 0) {
	  if (ratioi>=rmin && ratioi<=rmax) {
	    histos[0].first->Fill(ratioi,evWt);
	    histos[t_ieta].first->Fill(ratioi,evWt);
	  }
	  if (ratiof>=rmin && ratiof<=rmax) {
	    histos[0].second->Fill(ratiof,evWt);
	    histos[t_ieta].second->Fill(ratiof,evWt);
	  }
	}
      }
    }
  }

  //Fit the histograms
  TH1D *hbef1 = new TH1D("Eta1Bf","Mean vs i#eta",2*ietaMax,-ietaMax,ietaMax);
  TH1D *hbef2 = new TH1D("Eta2Bf","Median vs i#eta",2*ietaMax,-ietaMax,ietaMax);
  TH1D *haft1 = new TH1D("Eta1Af","Mean vs i#eta",2*ietaMax,-ietaMax,ietaMax);
  TH1D *haft2 = new TH1D("Eta2Af","Median vs i#eta",2*ietaMax,-ietaMax,ietaMax);
  for (int ieta=-ietaMax; ieta<=ietaMax; ++ieta) {
    int    bin   = (ieta < 0) ? (ieta+ietaMax+1) : (ieta+ietaMax);
    TH1D*  h1    = histos[ieta].first;
    double mean1 = h1->GetMean();
    double err1  = h1->GetMeanError();
    std::pair<double,double> fit1 = fitMean(h1,1);
    if (debug) {
      std::cout << ieta << " " << h1->GetName() << " " << mean1 << " +- " 
		<< err1 << " and " << fit1.first <<" +- " << fit1.second 
		<< std::endl;
    }
    if (ieta != 0) {
      hbef1->SetBinContent(bin,mean1);      hbef1->SetBinError(bin,err1);
      hbef2->SetBinContent(bin,fit1.first); hbef2->SetBinError(bin,fit1.second);
    }
    h1->Write();
    TH1D* h2 = histos[ieta].second;
    double mean2 = h2->GetMean();
    double err2  = h2->GetMeanError();
    std::pair<double,double> fit2 = fitMean(h2,1);
    if (debug) { 
      std::cout << ieta << " " << h2->GetName() << " " << mean2 << " +- " 
		<< err2 << " and " << fit2.first <<" +- " << fit2.second 
		<< std::endl;
    }
    if (ieta != 0) {
      haft1->SetBinContent(bin,mean2);      haft1->SetBinError(bin,err2);
      haft2->SetBinContent(bin,fit2.first); haft2->SetBinError(bin,fit2.second);
    }
    h2->Write();
  }
  fitPol0(hbef1,debug); fitPol0(hbef2,debug);
  fitPol0(haft1,debug); fitPol0(haft2,debug);
}

void CalibTree::fitPol0(TH1D* hist, bool debug) {

  hist->GetXaxis()->SetTitle("i#eta");
  hist->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
  hist->GetYaxis()->SetRangeUser(0.4,1.6);
  TFitResultPtr Fit = hist->Fit("pol0","+QRWLS");
  if (debug) {
    std::cout << "Fit to Pol0 to " << hist->GetTitle() << ": " 
	      << Fit->Value(0) << " +- " << Fit->FitResult::Error(0)
	      << std::endl;
  }
  hist->Write();
}

void CalibTree::highEtaFactors(int ietaMax, bool debug) {
  std::map<unsigned int,std::pair<double,double> >::const_iterator itr;
  std::pair<double,double> cfacp, cfacn;
  cfacp = cfacn = std::pair<double,double>(1.0,0.0);
  for (itr=Cprev.begin(); itr !=Cprev.end(); ++itr) {
    unsigned int detid = itr->first;
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detid, subdet, zside, ieta, iphi, depth);
    if ((ieta == ietaMax) && (depth == 1)) {
      if (zside > 0) cfacp = itr->second;
      else           cfacn = itr->second;
    }
  }
  if (debug) {
    std::cout << "Correction factor for (" << ietaMax << ",1) = " 
	      << cfacp.first << ":" << cfacp.second << " and (" 
	      << -ietaMax << ",1) = " << cfacn.first << ":" 
	      << cfacn.second << std::endl;
  }
  for (itr=Cprev.begin(); itr !=Cprev.end(); ++itr) {
    unsigned int detid = itr->first;
    int subdet, depth, zside, ieta, iphi;
    unpackDetId(detid, subdet, zside, ieta, iphi, depth);
    if (ieta > ietaMax) {
      Cprev[detid] = (zside > 0) ? cfacp : cfacn;
      if (debug) {
	std::cout << "Set correction factor for (" << zside*ieta
		  << "," << depth << ") = " << Cprev[detid].first
		  << ":" << Cprev[detid].second << std::endl;
      }
    }
  }      
}

CalibTree::energyCalor CalibTree::energyHcal(double pmom, 
					     const Long64_t& entry, 
					     bool final) {

  double etot  = t_eHcal;
  double etot2 = t_eHcal;
  double ediff = (t_eHcal30 - t_eHcal10);
  if (final) {
    etot = etot2 = 0;
    for (unsigned int idet=0; idet<(*t_DetIds).size(); idet++) { 
      if (selectPhi((*t_DetIds)[idet])) {
	unsigned int id = (*t_DetIds)[idet];
	double hitEn(0);
	unsigned int detid = truncateId(id,truncateFlag_,false);
	if (Cprev.find(detid) != Cprev.end()) 
	  hitEn = Cprev[detid].first * (*t_HitEnergies)[idet];
	else 
	  hitEn = (*t_HitEnergies)[idet];
	if (cFactor_) hitEn *= cFactor_->getCorr(t_Run,id);
	  etot  += hitEn;
	  etot2 += ((*t_HitEnergies)[idet]);
      }
    }
    // Now the outer cone 
    double etot1(0), etot3(0);
    if (t_DetIds1 != 0 && t_DetIds3 != 0) {
      for (unsigned int idet=0; idet<(*t_DetIds1).size(); idet++) { 
	if (selectPhi((*t_DetIds1)[idet])) {
	  unsigned int id    = (*t_DetIds1)[idet];
	  unsigned int detid = truncateId(id,truncateFlag_,false);
	  double hitEn(0);
	  if (Cprev.find(detid) != Cprev.end()) 
	    hitEn = Cprev[detid].first * (*t_HitEnergies1)[idet];
	  else 
	    hitEn = (*t_HitEnergies1)[idet];
	  if (cFactor_) hitEn *= cFactor_->getCorr(t_Run,id);
	  etot1  += hitEn;
	}
      }
      for (unsigned int idet=0; idet<(*t_DetIds3).size(); idet++) { 
	if (selectPhi((*t_DetIds3)[idet])) {
	  unsigned int id    = (*t_DetIds3)[idet];
	  unsigned int detid = truncateId(id,truncateFlag_,false);
	  double hitEn(0);
	  if (Cprev.find(detid) != Cprev.end()) 
	    hitEn = Cprev[detid].first * (*t_HitEnergies3)[idet];
	  else 
	    hitEn = (*t_HitEnergies3)[idet];
	  if (cFactor_) hitEn *= cFactor_->getCorr(t_Run,id);
	    etot3  += hitEn;
	}
      }
    }
    ediff = etot3-etot1;
  }
  // PU correction only for loose isolation cut
  double ehcal = (((rcorForm_ == 3) && (cFactor_ != nullptr)) ?
		  (etot * cFactor_->getCorr(entry)) : ((puCorr_ == 0) ? etot : 
		  ((puCorr_ < 0) ? (etot*puFactor(-puCorr_,t_ieta,pmom,etot,ediff)) :
		   puFactorRho(puCorr_,t_ieta,t_rhoh,etot))));
  return CalibTree::energyCalor(etot,etot2,ehcal);
}
