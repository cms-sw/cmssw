//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jan 29 14:18:27 2015 by ROOT version 5.34/19
// from TTree RecJet/RecJet Tree
// found on file: True.root
//////////////////////////////////////////////////////////

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TPaveStats.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>
#include <iostream>
#include <string>
#include <cmath>
#include <map>
#include <fstream>
#include <TF1.h>

#include <iomanip>
#include <sstream>

// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class RecJet {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           cells;
  Int_t           mysubd;
  Int_t           depth;
  Int_t           ieta;
  Int_t           iphi;
  Float_t         mom0_MB;
  Float_t         mom1_MB;
  Float_t         mom2_MB;
  Float_t         mom3_MB;
  Float_t         mom4_MB;
  Float_t         mom0_Diff;
  Float_t         mom1_Diff;
  Float_t         mom2_Diff;
  Float_t         mom3_Diff;
  Float_t         mom4_Diff;
  Int_t           trigbit;
  Double_t        rnnumber;

  // List of branches
  TBranch        *b_cells;     //!
  TBranch        *b_mysubd;    //!
  TBranch        *b_depth;     //!
  TBranch        *b_ieta;      //!
  TBranch        *b_iphi;      //!
  TBranch        *b_mom0_MB;   //!
  TBranch        *b_mom1_MB;   //!
  TBranch        *b_mom2_MB;   //!
  TBranch        *b_mom3_MB;   //!
  TBranch        *b_mom4_MB;   //!
  TBranch        *b_mom0_Diff;   //!
  TBranch        *b_mom1_Diff;   //!
  TBranch        *b_mom2_Diff;   //!
  TBranch        *b_mom3_Diff;   //!
  TBranch        *b_mom4_Diff; 
  TBranch        *b_trigbit;   //!
  TBranch        *b_rnnumber;  //!

  struct MyInfo {
    double Mom0, Mom1, Mom2, Mom3, Mom4;
    MyInfo() {
      Mom0 = Mom1 = Mom2 = Mom3 = Mom4 = 0.;
    }
  };

  struct CFactors {
    double cfac1, efac1, cfac2, efac2;
    CFactors() {
      cfac1 = cfac2 = 1;
      efac1 = efac2 = 0;
    }
  };

  struct Hists {
    TH1D *h1, *h2, *h3, *h4, *h5, *h6, *h7, *h8, *h9;
    Hists() {
      h1 = h2 = h3 = h4 = h5= h6= h7= h8= h9=0;
    }
  };
  TFile*                                           file;
  std::string                                      detector;
  int                                              mode_;
  bool                                             loadTrig_;
  std::map<unsigned int,MyInfo>                    mNoise_;
  std::map<std::pair<unsigned int,int>,MyInfo>     mTrig_;
  std::vector<unsigned int>                        dets_;
  double                                           factor_;
  double                                           err_mean, err_var;
  std::map<unsigned int,CFactors>                  corrFactor_;

  RecJet(std::string fname, int mode=0);
  virtual ~RecJet();
  bool             OpenFile(std::string fname);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  virtual Int_t    Cut(Long64_t entry);
  std::map<unsigned int,RecJet::MyInfo> LoopMap();
  virtual void     LoopNoise();
  virtual void     Loop(int subdet, std::string indx, bool clear);
  virtual void     LoopIter(int subdet, std::string indx, bool clear, int maxIter=100);
  virtual void     LoopIterate(int subdet, std::string indx, double emin, double emax, int maxIter=100);
    std::map<unsigned int,RecJet::Hists> MakeCorr(std::map<unsigned int,RecJet::MyInfo>&, int, int);
  virtual void     LoopMean(int subdet, std::string indx);
  virtual void     eta_distribution(std::string fname="Eta.root", bool var=true);
  virtual void     Alleta_distribution(std::string fname="AllEta.root", bool var=true, bool HBHE=true);
  virtual void     det_distribution(std::string fname="Det.root");
  virtual void     Fit(std::string rootfile="HCALNZS2015_Final_Pedestal_magnet_test.root", std::string textfile="fit.txt");
  std::pair<double,double> SubtractNoise(unsigned int, double, double, bool);
  virtual void     Disturb(int subd);
  std::map<unsigned int,RecJet::MyInfo> MultiplyEnergy(std::map<unsigned int,RecJet::MyInfo>&,int);
  virtual void     ChangeMoments(std::map<unsigned int,RecJet::MyInfo>::iterator&, double);
  std::map<unsigned int,RecJet::MyInfo> LoadNoise();
  virtual void     MeanVariance(std::map<unsigned int,RecJet::MyInfo>::iterator &mitr, std::pair<double,double>& mean, std::pair<double,double>& variance);
  std::vector<std::pair<double,double> > Staggered_CorrFactor(int subd, std::map<unsigned int,RecJet::Hists>::iterator &hitr, bool varmethod);
  std::vector<std::pair<double,double> > CorrFactor(int subd, std::map<unsigned int,RecJet::Hists>::iterator &hitr, bool varmethod);
  void StoreCorrFactor(const std::map<unsigned int,RecJet::Hists> &, std::map<unsigned int,RecJet::CFactors>&, bool clear=true);
  std::pair<double,double> MeanDeviation(int, std::map<unsigned int,RecJet::CFactors>&);
  void             WriteCorrFactor(std::string& outFile, std::string& infile, bool var=false);
  virtual void     Error_study(std::string fname, std::string det, int depth);
  virtual void     LoopMapTrig();
  virtual void     LoopTrig(int sdet, int eta, int phi, int dep, std::string indx);
  std::map<unsigned int,RecJet::MyInfo> BuildMap(int sdet, bool cfirst, double emin, double emax);
};

RecJet::RecJet(std::string fname, int mode) : fChain(0), file(0), mode_(mode),
					      loadTrig_(false), factor_(1.) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  std::cout << "Open file " << fname << std::endl;
  //  std::cout << "hi.." << std::endl;
  file = new TFile(fname.c_str());
  //file->cd("minbiasana");
  //  std::cout << "hi.." << std::endl;
  TTree *tree;
  //gDirectory->GetObject("RecJet", tree);
  file->GetObject("RecJet", tree);
  Init(tree);
}

RecJet::~RecJet() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

bool RecJet::OpenFile(std::string fname) {
  if (fChain) {
    std::cout << "Close File " << fChain->GetCurrentFile()->GetName() << std::endl;
    delete fChain->GetCurrentFile();
  }
  file = new TFile(fname.c_str());
  if (file) {
    file->cd("minbiasana");
    TTree *tree;
    gDirectory->GetObject("RecJet", tree);
    Init(tree);
    return true;
  } else {
    return false;
  }
}

Int_t RecJet::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t RecJet::LoadTree(Long64_t entry) {
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

void RecJet::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("cells", &cells, &b_cells);
  fChain->SetBranchAddress("mysubd", &mysubd, &b_mysubd);
  fChain->SetBranchAddress("depth", &depth, &b_depth);
  fChain->SetBranchAddress("ieta", &ieta, &b_ieta);
  fChain->SetBranchAddress("iphi", &iphi, &b_iphi);
  fChain->SetBranchAddress("mom0_MB", &mom0_MB, &b_mom0_MB);
  fChain->SetBranchAddress("mom1_MB", &mom1_MB, &b_mom1_MB);
  fChain->SetBranchAddress("mom2_MB", &mom2_MB, &b_mom2_MB);
  fChain->SetBranchAddress("mom3_MB", &mom3_MB, &b_mom3_MB);
  fChain->SetBranchAddress("mom4_MB", &mom4_MB, &b_mom4_MB);
  if (mode_ == 1) {
    fChain->SetBranchAddress("mom0_Diff", &mom0_Diff, &b_mom0_Diff);
    fChain->SetBranchAddress("mom1_Diff", &mom1_Diff, &b_mom1_Diff);
    fChain->SetBranchAddress("mom2_Diff", &mom2_Diff, &b_mom2_Diff);
    fChain->SetBranchAddress("mom3_Diff", &mom3_Diff, &b_mom3_Diff);
    fChain->SetBranchAddress("mom4_Diff", &mom4_Diff, &b_mom4_Diff);
  }
  fChain->SetBranchAddress("trigbit", &trigbit, &b_trigbit);
  fChain->SetBranchAddress("rnnumber", &rnnumber, &b_rnnumber);
  Notify();
}

Bool_t RecJet::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void RecJet::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t RecJet::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

std::map<unsigned int,RecJet::MyInfo> RecJet::LoopMap() {
  //   In a ROOT session, you can do:
  //      Root > .L RecJet.C
  //      Root > RecJet t
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

  std::map<unsigned int,RecJet::MyInfo> m_;
  if (fChain != 0) {
    Long64_t nentries = fChain->GetEntriesFast();
    Long64_t nbytes = 0, nb = 0;

    for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;

      nb = fChain->GetEntry(jentry);   nbytes += nb;
      unsigned int detId1 = ((mysubd<<20) | ((depth&0x1f)<<14) | 
			     ((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) | 
			     (iphi&0x7f));
      std::map<unsigned int, RecJet::MyInfo>::iterator mitr = m_.find(detId1);
      if (mitr == m_.end()) {
	RecJet::MyInfo info;
	m_[detId1] = info ;
	mitr = m_.find(detId1) ;
      }
      if (mode_ == 1) {
	mitr->second.Mom0 += mom0_Diff;
	mitr->second.Mom1 += mom1_Diff;
	mitr->second.Mom2 += mom2_Diff;
	mitr->second.Mom3 += mom3_Diff;
	mitr->second.Mom4 += mom4_Diff;
      } else {
	mitr->second.Mom0 += mom0_MB;
	mitr->second.Mom1 += mom1_MB;
	mitr->second.Mom2 += mom2_MB;
	mitr->second.Mom3 += mom3_MB;
	mitr->second.Mom4 += mom4_MB;
      }
    }
  } 
  return m_;
}

void RecJet::LoopNoise() {

  mNoise_ = LoopMap();
  if (mNoise_.size() == 0) return;
  TH1D* h[4];
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  char fname[80], name[80], title[80];
  for (int subd=0; subd<4; ++subd) {
    sprintf (fname, "Pedestal_2015D_%s.root", det[subd].c_str());
    TFile f(fname, "recreate");
    detector = det[subd];
    ofstream myfile;
    sprintf (fname,"%s_Noise.txt",detector.c_str());
    myfile.open(fname);
    sprintf(name,"%s",det[subd].c_str());
    sprintf(title,"Energy Distribution for %s",det[subd].c_str());
    if (subd !=3) h[subd] = new TH1D(name,title,100,-1.,1.);
    else          h[subd] = new TH1D(name,title,100,0.,5.);
    std::map<unsigned int,RecJet::Hists> hh_ ;
    for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = mNoise_.begin();
	 mitr !=mNoise_.end(); mitr++) {
      unsigned int sdet    = ((mitr->first)>>20)&0x7;
      if (sdet == (unsigned int)(subd+1)) {
	unsigned int detId2  = (mitr->first)&0x7fff80;
	std::map<unsigned int,RecJet::Hists>::iterator hitr = hh_.find(detId2);
	int keta = (((mitr->first)&0x2000) ? (((mitr->first)>>7)&0x3f) :
		    -(((mitr->first)>>7)&0x3f));
	int dept = (((mitr->first)>>14)&0x1f);
	int kphi = ((mitr->first)&0x7f);
	if (hitr == hh_.end()) {
	  RecJet::Hists hh;
	  hh_[detId2]  = hh;
	  hitr = hh_.find(detId2);
	  sprintf (name,  "hist%s%d%d", detector.c_str(), keta, dept);
	  sprintf (title, "#phi (%s #eta %d depth %d)", detector.c_str(), keta, dept);
	  hitr->second.h1 = new TH1D(name, title, 72, 0, 72.);
	  hitr->second.h1->GetXaxis()->SetTitle("i#phi"); 
	  hitr->second.h1->GetXaxis()->CenterTitle();
	  hitr->second.h1->GetYaxis()->SetTitle("Mean Energy (GeV)"); 
	  hitr->second.h1->GetYaxis()->CenterTitle();
	  sprintf (name,  "histv%s%d%d", detector.c_str(), keta, dept);
	  hitr->second.h2 = new TH1D(name, title, 72, 0, 72.);
	  hitr->second.h2->GetXaxis()->SetTitle("i#phi"); 
	  hitr->second.h2->GetXaxis()->CenterTitle();
	  hitr->second.h2->GetYaxis()->SetTitle("Variance (GeV^2)");  
	  hitr->second.h2->GetYaxis()->CenterTitle();
	}
	std::pair<double,double> mean, variance;
	MeanVariance(mitr,mean,variance);
        h[subd]->Fill(mean.first);
	hitr->second.h1->SetBinContent(kphi,mean.first);
	hitr->second.h1->SetBinError(kphi,mean.second);
	hitr->second.h2->SetBinContent(kphi,variance.first);
	hitr->second.h2->SetBinError(kphi,variance.second);
	myfile << mitr->first << "\t" << mitr->second.Mom0 << "\t" 
	       << mitr->second.Mom1 << "\t" << mitr->second.Mom2 << "\t"
	       << mitr->second.Mom3 << "\t" << mitr->second.Mom4 << std::endl;
      }
    }
    h[subd]->Write();
    myfile.close();
    for (std::map<unsigned int,RecJet::Hists>::iterator hitr = hh_.begin();
	 hitr != hh_.end(); ++hitr) {
      int keta = (((hitr->first)&0x2000) ? (((hitr->first)>>7)&0x3f) :
		    -(((hitr->first)>>7)&0x3f));
      int dept = (((hitr->first)>>14)&0x1f);
      sprintf (fname, "correction_mean%s%d%d", detector.c_str(), keta, dept);
      sprintf (title, "Correction Factor (Mean) for (%s #eta %d depth %d)", detector.c_str(), keta, dept);
      hitr->second.h3 = new TH1D(fname, title, 72, 0., 72.);
      hitr->second.h3->GetXaxis()->SetTitle("i#phi"); 
      hitr->second.h3->GetXaxis()->CenterTitle();
      hitr->second.h3->GetYaxis()->SetTitle("Correction Factor"); 
      hitr->second.h3->GetYaxis()->CenterTitle();
      sprintf (fname, "correction_variance%s%d%d", detector.c_str(), keta,dept);
      sprintf (title, "Correction Factor (Variance) for (%s #eta %d depth %d)", detector.c_str(), keta, dept);
      hitr->second.h4 = new TH1D(fname, title, 72, 0., 72.);
      hitr->second.h4->GetXaxis()->SetTitle("i#phi"); 
      hitr->second.h4->GetXaxis()->CenterTitle();
      hitr->second.h4->GetYaxis()->SetTitle("Correction Factor"); 
      hitr->second.h4->GetYaxis()->CenterTitle();
      sprintf (fname, "correction_variance_mine%s%d%d", detector.c_str(), keta,dept);
      sprintf (title, "Correction Factor (Variance) for (%s #eta %d depth %d)", detector.c_str(), keta, dept);
      hitr->second.h8 = new TH1D(fname, title, 72, 0., 72.);
      hitr->second.h8->GetXaxis()->SetTitle("i#phi");
      hitr->second.h8->GetXaxis()->CenterTitle();
      hitr->second.h8->GetYaxis()->SetTitle("Corrected Variance");
      hitr->second.h8->GetYaxis()->CenterTitle();
      sprintf (fname, "correction_mean_mine%s%d%d", detector.c_str(), keta,dept);
      sprintf (title, "Correction Factor (Mean) for (%s #eta %d depth %d)", detector.c_str(), keta, dept);
      hitr->second.h9 = new TH1D(fname, title, 72, 0., 72.);
      hitr->second.h9->GetXaxis()->SetTitle("i#phi");
      hitr->second.h9->GetXaxis()->CenterTitle();
      hitr->second.h9->GetYaxis()->SetTitle("Corrected Mean");
      hitr->second.h9->GetYaxis()->CenterTitle();
      std::vector<std::pair<double,double> > corr = CorrFactor(subd, hitr, false);
      for (unsigned int i=0; i<corr.size(); ++i) {
	if (corr[i].first > 0){
	  hitr->second.h3->SetBinContent(i+1, corr[i].first);
	  hitr->second.h3->SetBinError(i+1,   corr[i].second);
	}
      }                 
      corr = CorrFactor(subd, hitr, true);
      for (unsigned int i=0; i<corr.size(); ++i) {
	if (corr[i].first > 0){
	  hitr->second.h4->SetBinContent(i+1, corr[i].first);
	  hitr->second.h4->SetBinError(i+1,   corr[i].second);
	}
      }                 
      hitr->second.h1->Write();
      hitr->second.h2->Write();
      hitr->second.h3->Write();
      hitr->second.h4->Write();
      hitr->second.h8->Write();
      hitr->second.h9->Write();

    }
    f.Close();
  }
}

void RecJet::Loop(int sdet, std::string indx, bool clear) {

  std::map<unsigned int,RecJet::MyInfo> m_ = LoopMap();

  if (m_.size() == 0) return;

  char fname[80], name[80], title[80];
  m_ = MultiplyEnergy(m_,0);

  int subd = ((sdet>=1 && sdet<=4) ? (sdet-1) : 0);
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  sprintf (fname, "HCALNZS2015%s_phiRecoDistribution_Final%s.root", indx.c_str(), det[subd].c_str());
  TFile f(fname, "recreate");
  detector = det[subd];
  sprintf(name,"%s", det[subd].c_str());
  sprintf(title,"Energy Distribution for %s", det[subd].c_str());
  std::map<unsigned int,RecJet::Hists> hh_ = MakeCorr(m_, subd, -1);

  StoreCorrFactor(hh_, corrFactor_, clear);
  std::pair<double,double> meandev = MeanDeviation(sdet, corrFactor_);
  std::cout << "Mean deviation from 1: " << meandev.first << " and " 
	    << meandev.second << std::endl;
  sprintf(name, "Comparison%s", detector.c_str());
  sprintf(title,"Comparison of Correction Factor from Variance & Mean (%s)", detector.c_str());
  TH2D* h = new TH2D(name, title, 100, 0.5, 2.0, 100., 0.5, 2.0);
  for (std::map<unsigned int,RecJet::Hists>::const_iterator hitr = hh_.begin();
       hitr != hh_.end(); ++hitr) {
    if (hitr->second.h3 != 0 && hitr->second.h4 != 0) {
      for (int i=1; i<=hitr->second.h3->GetNbinsX(); ++i) {
	double x1 = hitr->second.h3->GetBinContent(i);
	double x2 = hitr->second.h4->GetBinContent(i);
        if (x1 != 0 && x2 != 0) h->Fill(x1,x2);
      }
    }
    if (hitr->second.h1 != 0) hitr->second.h1->Write();
    if (hitr->second.h2 != 0) hitr->second.h2->Write();
    if (hitr->second.h3 != 0) hitr->second.h3->Write();
    if (hitr->second.h4 != 0) hitr->second.h4->Write();
    if (hitr->second.h5 != 0) hitr->second.h5->Write();
    if (hitr->second.h6 != 0) hitr->second.h6->Write();
    if (hitr->second.h7 != 0) hitr->second.h7->Write();
    if (hitr->second.h8 != 0) hitr->second.h8->Write();
    if (hitr->second.h9 != 0) hitr->second.h9->Write();


  }
  h->Write();
  f.Close();
}

void RecJet::LoopIter(int sdet, std::string indx, bool clear, int maxIter) {

  std::map<unsigned int,RecJet::MyInfo> m_ = LoopMap();
  if (m_.size() == 0) return;

  int subd = ((sdet>=1 && sdet<=4) ? (sdet-1) : 0);
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  char fname[80], name[80], title[80];
  sprintf (fname, "Pileup%s_phiRecoDistribution_%s.root", indx.c_str(), det[subd].c_str());
  TFile f(fname, "recreate");
  detector = det[subd];
  sprintf (name, "Conv1%s", detector.c_str());
  sprintf (title, "Mean Deviation vs Iteration (%s)", detector.c_str());
  TH1D* h1 = new TH1D(name, title, maxIter+1, 0, maxIter+1);
  sprintf (name, "Conv2%s", detector.c_str());
  TH1D* h2 = new TH1D(name, title, maxIter+1, 0, maxIter+1);
  // 0th iteration and initiate correction factor  
  std::map<unsigned int,RecJet::Hists> hh_ = MakeCorr(m_, subd, 0);
  StoreCorrFactor(hh_, corrFactor_, clear);
  std::pair<double,double> meandev = MeanDeviation(sdet, corrFactor_);
  h1->SetBinContent(0,meandev.first);
  h2->SetBinContent(0,meandev.second);
  std::cout << "Mean deviation from 1 in iteration 0: " << meandev.first
	    << " and " << meandev.second << std::endl;
  for (std::map<unsigned int,RecJet::Hists>::const_iterator hitr = hh_.begin();
       hitr != hh_.end(); ++hitr) {
    if (hitr->second.h1 != 0) hitr->second.h1->Write();
    if (hitr->second.h2 != 0) hitr->second.h2->Write();
    if (hitr->second.h3 != 0) hitr->second.h3->Write();
    if (hitr->second.h4 != 0) hitr->second.h4->Write();
    if (hitr->second.h5 != 0) hitr->second.h5->Write();
    if (hitr->second.h6 != 0) hitr->second.h6->Write();
    if (hitr->second.h7 != 0) hitr->second.h7->Write();
    if (hitr->second.h8 != 0) hitr->second.h8->Write();
    if (hitr->second.h9 != 0) hitr->second.h9->Write();

  }
  //Now iterate
  for (int kit = 1; kit <= maxIter; ++kit) {
    std::map<unsigned int,RecJet::MyInfo>   m2_  = MultiplyEnergy(m_,1);
    std::map<unsigned int,RecJet::Hists>    hh2_ = MakeCorr(m_, subd, kit);
    std::map<unsigned int,RecJet::CFactors> cF2_;
    StoreCorrFactor(hh2_, cF2_, true);
    meandev = MeanDeviation(sdet, cF2_);
    h1->SetBinContent(kit,meandev.first);
    h2->SetBinContent(kit,meandev.second);
    std::cout << "Mean deviation from 1 in iteration " << kit << ": " 
	      << meandev.first << " and " << meandev.second << std::endl;
    //Multiply cF2 to CorrFactor and make exit route if meandev is small
    for (std::map<unsigned int,RecJet::CFactors>::iterator it = corrFactor_.begin();
       it != corrFactor_.end(); ++it) {
      std::map<unsigned int,RecJet::CFactors>::iterator jt = cF2_.find(it->first);
      if (jt != cF2_.end()) {
	it->second.cfac1 *= (jt->second.cfac1);
	it->second.efac1 *= (jt->second.cfac1);
	it->second.cfac2 *= (jt->second.cfac2);
	it->second.efac2 *= (jt->second.cfac2);
      }
    }
    if (meandev.first < 0.0001 || kit==maxIter) {
      for (std::map<unsigned int,RecJet::Hists>::const_iterator hitr = hh2_.begin();
	   hitr != hh2_.end(); ++hitr) {
	if (hitr->second.h1 != 0) hitr->second.h1->Write();
	if (hitr->second.h2 != 0) hitr->second.h2->Write();
	if (hitr->second.h3 != 0) hitr->second.h3->Write();
	if (hitr->second.h4 != 0) hitr->second.h4->Write();
	if (hitr->second.h5 != 0) hitr->second.h5->Write();
	if (hitr->second.h6 != 0) hitr->second.h6->Write();
	if (hitr->second.h7 != 0) hitr->second.h7->Write();
	if (hitr->second.h8 != 0) hitr->second.h8->Write();
	if (hitr->second.h9 != 0) hitr->second.h9->Write();

      }
      break;
    }
  }

  sprintf (name, "Comparison%s", detector.c_str());
  sprintf (title,"Comparison of Correction Factor from Variance & Mean (%s)", detector.c_str());
  TH2D* h = new TH2D(name, title, 100, 0.5, 2.0, 100, 0.5, 2.0);
  for (std::map<unsigned int,RecJet::CFactors>::iterator itr = corrFactor_.begin();
       itr != corrFactor_.end(); ++itr) {
    double x1 = itr->second.cfac1;
    double x2 = itr->second.cfac2;
    if (x1 != 0 && x2 != 0) h->Fill(x1,x2);
  }
  h->Write();
  h1->Write();
  h2->Write();
  f.Close();
}

    /*h1->distribution of mean energy
      h2->distribution of variance
      h3->correction factor from variance method
      h4->correction factor from mean method
      h5->corrected mean by variance method
      h6->corrected mean from mean method
      h7->corrected variance
      h8->corrected variance directly from the mean of variance
      h9->corrected mean directly from the mean of mean energy*/


void RecJet::LoopIterate(int sdet, std::string indx, double emin, double emax, int maxIter) {

  int subd = ((sdet>=1 && sdet<=4) ? (sdet-1) : 0);
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  char fname[80], name[80], title[80];
  sprintf (fname, "Pileup%s_phiRecoDistribution_%s.root", indx.c_str(), det[subd].c_str());
  TFile f(fname, "recreate");
  detector = det[subd];
  sprintf (name, "Conv1%s", detector.c_str());
  sprintf (title, "Mean Deviation vs Iteration (%s)", detector.c_str());
  TH1D* h1 = new TH1D(name, title, maxIter+1, 0, maxIter+1);
  sprintf (name, "Conv2%s", detector.c_str());
  TH1D* h2 = new TH1D(name, title, maxIter+1, 0, maxIter+1);
  // 0th iteration and initiate correction factor
  std::map<unsigned int,RecJet::MyInfo> m_ = BuildMap(sdet, true, emin, emax);
  std::map<unsigned int,RecJet::Hists> hh_ = MakeCorr(m_, subd, 0);
  StoreCorrFactor(hh_, corrFactor_, true);
  std::pair<double,double> meandev = MeanDeviation(sdet, corrFactor_);
  h1->SetBinContent(0,meandev.first);
  h2->SetBinContent(0,meandev.second);
  std::cout << "Mean deviation from 1 in iteration 0: " << meandev.first
	    << " and " << meandev.second << std::endl;
  for (std::map<unsigned int,RecJet::Hists>::const_iterator hitr = hh_.begin();
       hitr != hh_.end(); ++hitr) {
    if (hitr->second.h1 != 0) hitr->second.h1->Write();
    if (hitr->second.h2 != 0) hitr->second.h2->Write();
    if (hitr->second.h3 != 0) hitr->second.h3->Write();
    if (hitr->second.h4 != 0) hitr->second.h4->Write();
    if (hitr->second.h5 != 0) hitr->second.h5->Write();
    if (hitr->second.h6 != 0) hitr->second.h6->Write();
    if (hitr->second.h7 != 0) hitr->second.h7->Write();
    if (hitr->second.h8 != 0) hitr->second.h8->Write();
    if (hitr->second.h9 != 0) hitr->second.h9->Write();

  }
  //Now iterate
  for (int kit = 1; kit <= maxIter; ++kit) {
    std::map<unsigned int,RecJet::MyInfo>   m2_  = BuildMap(sdet, false, emin, emax);
    std::map<unsigned int,RecJet::Hists>    hh2_ = MakeCorr(m_, subd, kit);
    std::map<unsigned int,RecJet::CFactors> cF2_;
    StoreCorrFactor(hh2_, cF2_, true);
    meandev = MeanDeviation(sdet, cF2_);
    h1->SetBinContent(kit,meandev.first);
    h2->SetBinContent(kit,meandev.second);
    std::cout << "Mean deviation from 1 in iteration " << kit << ": " 
	      << meandev.first << " and " << meandev.second << std::endl;
    //Multiply cF2 to CorrFactor and make exit route if meandev is small
    for (std::map<unsigned int,RecJet::CFactors>::iterator it = corrFactor_.begin();
       it != corrFactor_.end(); ++it) {
      std::map<unsigned int,RecJet::CFactors>::iterator jt = cF2_.find(it->first);
      if (jt != cF2_.end()) {
	it->second.cfac1 *= (jt->second.cfac1);
	it->second.efac1 *= (jt->second.cfac1);
	it->second.cfac2 *= (jt->second.cfac2);
	it->second.efac2 *= (jt->second.cfac2);
      }
    }
    if (meandev.first < 0.0001 || kit==maxIter) {
      for (std::map<unsigned int,RecJet::Hists>::const_iterator hitr = hh2_.begin();
	   hitr != hh2_.end(); ++hitr) {
	if (hitr->second.h1 != 0) hitr->second.h1->Write();
	if (hitr->second.h2 != 0) hitr->second.h2->Write();
	if (hitr->second.h3 != 0) hitr->second.h3->Write();
	if (hitr->second.h4 != 0) hitr->second.h4->Write();
	if (hitr->second.h5 != 0) hitr->second.h5->Write();
	if (hitr->second.h6 != 0) hitr->second.h6->Write();
	if (hitr->second.h7 != 0) hitr->second.h7->Write();
	if (hitr->second.h8 != 0) hitr->second.h8->Write();
	if (hitr->second.h9 != 0) hitr->second.h9->Write();

      }
      break;
    }
  }

  sprintf (name, "Comparison%s", detector.c_str());
  sprintf (title,"Comparison of Correction Factor from Variance & Mean (%s)", detector.c_str());
  TH2D* h = new TH2D(name, title, 100, 0.5, 2.0, 100, 0.5, 2.0);
  for (std::map<unsigned int,RecJet::CFactors>::iterator itr = corrFactor_.begin();
       itr != corrFactor_.end(); ++itr) {
    double x1 = itr->second.cfac1;
    double x2 = itr->second.cfac2;
    if (x1 != 0 && x2 != 0) h->Fill(x1,x2);
  }
  h->Write();
  h1->Write();
  h2->Write();
  f.Close();
}

std::map<unsigned int, RecJet::Hists> 
RecJet::MakeCorr(std::map<unsigned int,RecJet::MyInfo>& m_, int subd, 
		 int itr) {

  char name[80], title[80];
  std::map<unsigned int, RecJet::Hists> hh_ ;
  for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.begin();
       mitr !=m_.end(); mitr++) {
    unsigned int sdet0  = ((mitr->first)>>20)&0x7;
    if (sdet0 == (unsigned int)(subd+1)) {
      unsigned int detId2  = (mitr->first)&0x7fff80 ;
      std::map<unsigned int,RecJet::Hists>::iterator hitr = hh_.find(detId2);
      if (hitr == hh_.end()) {
	RecJet::Hists hh;
	hh_[detId2] = hh;
	hitr = hh_.find(detId2);
	int keta = (((mitr->first)&0x2000) ? (((mitr->first)>>7)&0x3f) :
		    -(((mitr->first)>>7)&0x3f));
	int dept = (((mitr->first)>>14)&0x1f);
	if (itr >= 0) {
	  sprintf (name,  "hist%s%d %d%d", detector.c_str(), keta, dept, itr);
	  sprintf (title, "#phi (%s #eta %d depth %d iter %d)", detector.c_str(), keta, dept, itr);
	} else {
	  sprintf (name,  "hist%s%d%d", detector.c_str(), keta, dept);
	  sprintf (title, "#phi (%s #eta %d depth %d)", detector.c_str(), keta, dept);
	}
	hitr->second.h1 = new TH1D(name, title, 72, 0, 72.);
        hitr->second.h1->GetXaxis()->SetTitle("i#phi");
	hitr->second.h1->GetXaxis()->CenterTitle();
        hitr->second.h1->GetYaxis()->SetTitle("Mean Energy(GeV)"); 
	hitr->second.h1->GetYaxis()->CenterTitle();
	if (itr >= 0) {
	  sprintf (name,  "histv%s%d %d%d", detector.c_str(), keta, dept, itr);
	} else {
	  sprintf (name,  "histv%s%d%d", detector.c_str(), keta, dept);
	}
	hitr->second.h2 = new TH1D(name, title, 72, 0, 72.);
	hitr->second.h2->GetXaxis()->SetTitle("i#phi"); 
	hitr->second.h2->GetXaxis()->CenterTitle();
	hitr->second.h2->GetYaxis()->SetTitle("Variance (GeV^2)"); 
	hitr->second.h2->GetYaxis()->CenterTitle();
      }
      std::pair<double,double> mean, variance;
      MeanVariance(mitr,mean,variance);
      std::pair<double,double> nsub = SubtractNoise(mitr->first,mean.first,
						    mean.second,false);
      int kphi = ((mitr->first)&0x7f);
      hitr->second.h1->SetBinContent(kphi,nsub.first);
      hitr->second.h1->SetBinError(kphi,nsub.second);
      nsub = SubtractNoise(mitr->first,variance.first,
			   variance.second,true);
      hitr->second.h2->SetBinContent(kphi,variance.first);
      hitr->second.h2->SetBinError(kphi,variance.second);
    }
  }
  for (std::map<unsigned int, RecJet::Hists>::iterator hitr = hh_.begin();
       hitr != hh_.end(); ++hitr) {
    int keta = (((hitr->first)&0x2000) ? (((hitr->first)>>7)&0x3f) :
		-(((hitr->first)>>7)&0x3f));
    int dept = (((hitr->first)>>14)&0x1f);
    if (itr >= 0) {
      sprintf (name,  "correctionMean%s%d %d%d", detector.c_str(), keta, dept, itr);
      sprintf (title, "Correction Factor (Mean) for (%s #eta %d depth %d iter %d)", detector.c_str(), keta, dept, itr);
    } else {
      sprintf (name,  "correctionMean%s%d%d", detector.c_str(), keta, dept);
      sprintf (title, "Correction Factor (Mean) for (%s #eta %d depth %d)", detector.c_str(), keta, dept);
    }
    hitr->second.h3 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h3->GetXaxis()->SetTitle("i#phi"); 
    hitr->second.h3->GetXaxis()->CenterTitle();
    hitr->second.h3->GetYaxis()->SetTitle("Correction Factor"); 
    hitr->second.h3->GetYaxis()->CenterTitle();
    if (itr >= 0) {
      sprintf (name,  "correctionVariance%s%d %d%d", detector.c_str(), keta, dept, itr);
      sprintf (title, "Correction Factor (Variance) for (%s #eta %d depth %d iter %d)", detector.c_str(), keta, dept, itr);
    } else {
      sprintf (name,  "correctionVariance%s%d%d", detector.c_str(), keta, dept);
      sprintf (title, "Correction Factor (Variance) for (%s #eta %d depth %d)", detector.c_str(), keta, dept);
    }
    hitr->second.h4 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h4->GetXaxis()->SetTitle("i#phi"); 
    hitr->second.h4->GetXaxis()->CenterTitle();
    hitr->second.h4->GetYaxis()->SetTitle("Correction Factor"); 
    hitr->second.h4->GetYaxis()->CenterTitle();
    sprintf (name,  "Corrected_Mean_var%s%d %d%d", detector.c_str(), keta, dept,itr);
    sprintf (title, "Corrected_Mean (by Variance) for (%s #eta %d depth %d) %d", detector.c_str(), keta, dept,itr);
    hitr->second.h5 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h5->GetXaxis()->SetTitle("i#phi"); 
    hitr->second.h5->GetXaxis()->CenterTitle();
    hitr->second.h5->GetYaxis()->SetTitle("Corrected_Mean"); 
    hitr->second.h5->GetYaxis()->CenterTitle();
    sprintf (name,  "Corrected_Mean_mean%s%d %d%d", detector.c_str(), keta, dept,itr);
    sprintf (title, "Corrected_Mean (by Mean) for (%s #eta %d depth %d) %d", detector.c_str(), keta, dept,itr);
    hitr->second.h6 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h6->GetXaxis()->SetTitle("i#phi");
    hitr->second.h6->GetXaxis()->CenterTitle();
    hitr->second.h6->GetYaxis()->SetTitle("Corrected_Mean");
    hitr->second.h6->GetYaxis()->CenterTitle();
    sprintf (name,  "Corrected_Variance%s%d %d%d", detector.c_str(), keta, dept,itr);
    sprintf (title, "Corrected_Variance (by Variance) for (%s #eta %d depth %d) %d", detector.c_str(), keta, dept,itr);
    hitr->second.h7 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h7->GetXaxis()->SetTitle("i#phi");
    hitr->second.h7->GetXaxis()->CenterTitle();
    hitr->second.h7->GetYaxis()->SetTitle("Corrected_Var");
    hitr->second.h7->GetYaxis()->CenterTitle();
    sprintf (name,  "Corrected_Variance_(my method)%s%d %d%d", detector.c_str(), keta, dept,itr);
    sprintf (title, "Corrected_Variance (by Variance) for (%s #eta %d depth %d) %d", detector.c_str(), keta, dept,itr);
    hitr->second.h8 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h8->GetXaxis()->SetTitle("i#phi");
    hitr->second.h8->GetXaxis()->CenterTitle();
    hitr->second.h8->GetYaxis()->SetTitle("Corrected_Var");
    hitr->second.h8->GetYaxis()->CenterTitle();
    sprintf (name,  "Corrected_Mean_(my method)%s%d %d%d", detector.c_str(), keta, dept,itr);
    sprintf (title, "Corrected_Variance (by Variance) for (%s #eta %d depth %d) %d", detector.c_str(), keta, dept,itr);
    hitr->second.h9 = new TH1D(name, title, 72, 0., 72.);
    hitr->second.h9->GetXaxis()->SetTitle("i#phi");
    hitr->second.h9->GetXaxis()->CenterTitle();
    hitr->second.h9->GetYaxis()->SetTitle("Corrected_Mean");
    hitr->second.h9->GetYaxis()->CenterTitle();

    std::vector<std::pair<double,double> > corr1 = CorrFactor(subd,hitr,false);
    for (unsigned int i=0; i<corr1.size(); ++i) {
      if (corr1[i].first != 0 && corr1[i].second !=0){
	hitr->second.h3->SetBinContent(i+1, corr1[i].first);
	hitr->second.h3->SetBinError(i+1,   corr1[i].second);
	double bin_cont = hitr->second.h1->GetBinContent(i+1) * corr1[i].first;
        double efac1 = (hitr->second.h1->GetBinError(i+1))*(corr1[i].first);
        double efac2 = (hitr->second.h1->GetBinContent(i+1))*(corr1[i].second);
        double bin_err = std::sqrt(efac1*efac1+efac2*efac2);
        if (bin_cont !=0) {
	  hitr->second.h6->SetBinContent(i+1, bin_cont);
	  hitr->second.h6->SetBinError(i+1, bin_err);
	}
      }
    }
    std::vector<std::pair<double,double> > corr2 = CorrFactor(subd,hitr, true);
    for (unsigned int i=0; i<corr2.size(); ++i) {
      if (corr2[i].first != 0 && corr2[i].second !=0) {
	double var = hitr->second.h2->GetBinContent(i+1)* corr2[i].first * corr2[i].first;
	hitr->second.h7->SetBinContent(i+1,var);
	double var_err1 = corr2[i].first * corr2[i].first * hitr->second.h2->GetBinError(i+1);
	double var_err2 = 2. * corr2[i].first * corr2[i].second * hitr->second.h2->GetBinContent(i+1);
	double var_err = std::sqrt(var_err1*var_err1 + var_err2*var_err2);
	hitr->second.h7->SetBinError(i+1,var_err);
	hitr->second.h4->SetBinContent(i+1, corr2[i].first);
	hitr->second.h4->SetBinError(i+1,   corr2[i].second);
	double bin_cont = hitr->second.h1->GetBinContent(i+1) * corr2[i].first;
        if(bin_cont !=0) hitr->second.h5->SetBinContent(i+1, bin_cont);
	double efac1 = (hitr->second.h1->GetBinError(i+1))*(corr2[i].first);
	double efac2 = (hitr->second.h1->GetBinContent(i+1))*(corr2[i].second);
	double bin_err = std::sqrt(efac1*efac1+efac2*efac2); 
        if(bin_cont !=0) hitr->second.h5->SetBinError(i+1, bin_err);
      }
    } 
  }
  return hh_;
}
 
void RecJet::LoopMean(int sdet, std::string indx) {
  std::map<unsigned int,RecJet::MyInfo> m_ = LoopMap();
  if (m_.size() == 0) return;
  int subd = ((sdet>=1 && sdet<=4) ? (sdet-1) : 0);
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  char fname[80],name[80],title[80];
  sprintf (fname, "True_etaRecoDistribution%s_%s.root", indx.c_str(), det[subd].c_str());
  TFile f(fname, "recreate");
  detector = det[subd];
 
  std::map<unsigned int,std::pair<TH1D*,TH1D*> > myMap;
  for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.begin();
       mitr !=m_.end(); mitr++) {
    unsigned int sdet0  = ((mitr->first)>>20)&0x7;
    if (sdet0 == (unsigned int)(subd+1)) {
      unsigned int detId2 =  (mitr->first)&0x7C000 ;
      std::map<unsigned int, std::pair<TH1D*,TH1D*> >::iterator hitr = myMap.find(detId2);
      int b2 = (subd == 3) ? 42 : ((subd == 1) ? 30 : 17);
      if (hitr == myMap.end()) {
	int dept = (((mitr->first)>>14)&0x1f);
	sprintf (name,  "hist%s%d", detector.c_str(), dept);
	sprintf (title, "Mean Response (%s depth %d)", detector.c_str(), dept);
	TH1D* h1 = new TH1D(name, title, 2*b2, -(double)(b2), (double)(b2));
	h1->GetXaxis()->SetTitle("i#eta"); h1->GetXaxis()->CenterTitle();
	sprintf (name,  "histv%s%d", detector.c_str(), dept);
	TH1D* h2 = new TH1D(name, title, 2*b2, -(double)(b2), (double)(b2));
	h2->GetXaxis()->SetTitle("i#eta"); h2->GetXaxis()->CenterTitle();
	if (subd == 3) {
	  h1->GetYaxis()->SetTitle("Mean # PE"); h2->GetYaxis()->SetTitle("Variace(# PE)");
	} else {
	  h1->GetYaxis()->SetTitle("Mean Energy (GeV)"); h2->GetYaxis()->SetTitle("Variance (GeV^2)");
	}
	myMap[detId2] = std::pair<TH1D*,TH1D*>(h1,h2); 
	hitr = myMap.find(detId2);
      }
      int keta        = (((mitr->first)&0x2000) ? (((mitr->first)>>7)&0x3f) :
			 -(((mitr->first)>>7)&0x3f));
      int bin         = keta + b2 + 1;
      std::pair<double,double> mean, variance;
      MeanVariance(mitr,mean,variance);
      std::pair<double,double> nsub = SubtractNoise(mitr->first,mean.first,mean.second,false);
      hitr->second.first->SetBinContent(bin,nsub.first);
      hitr->second.first->SetBinError(bin,nsub.second);
      nsub = SubtractNoise(mitr->first,variance.first,variance.second,true);
      hitr->second.second->SetBinContent(bin,nsub.first);
      hitr->second.second->SetBinError(bin,nsub.second);
    }
  }
  for (std::map<unsigned int, std::pair<TH1D*,TH1D*> >::iterator hitr = myMap.begin();
       hitr != myMap.end(); ++hitr) {
    hitr->second.first->Write(); hitr->second.second->Write();
  }
  f.Close();
}

void RecJet::Alleta_distribution(std::string fname, bool var, bool HBHE) {
  std::map<unsigned int,RecJet::MyInfo> m_ = LoopMap();
  TFile f(fname.c_str(), "recreate");
  if (m_.size() == 0) return;
  TH1D* h;
  char name[50], title[50];
  if (var) {
    sprintf(name,"Alleta");
    sprintf(title,"Variance distribution w.r.t.eta");
  } else {
    sprintf(name,"Alleta");
    sprintf(title,"Energy distribution w.r.t.eta");
  }
  h = new TH1D(name,title,82, -41., 41.);
  if (var) {
    h->GetYaxis()->SetTitle("Variance in Gev^2"); h->GetYaxis()->CenterTitle();
  } else {
    h->GetYaxis()->SetTitle("Mean Energy in Gev"); h->GetYaxis()->CenterTitle();
  }
  h->GetXaxis()->SetTitle("i#eta"); h->GetXaxis()->CenterTitle();


  std::map<int, std::pair<double,double> > h_eta;
  for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.begin();
       mitr !=m_.end(); mitr++) {
    int eta = (mitr->first)&0x2000 ? ((mitr->first)>>7)&0x3f : -(((mitr->first)>>7)&0x3f);
    std::map<int,std::pair<double,double> >::iterator hitr = h_eta.find(eta);
    if (hitr == h_eta.end()) {
      double energy=0.;
      double error =0.;
      h_eta[eta] = std::pair<double,double>(energy,error);
      hitr          = h_eta.find(eta);
    }
    std::pair<double,double> mean, variance;
    MeanVariance(mitr,mean,variance);
    std::pair<double, double> nsub;
    if (var) {
      nsub = SubtractNoise(mitr->first,variance.first,variance.second,true);
    } else {
     nsub = SubtractNoise(mitr->first,mean.first,mean.second,false);
    }
    hitr->second.first +=nsub.first;
    hitr->second.second +=pow(nsub.second,2);
  }
  for(std::map<int,std::pair<double,double> >::iterator hitr = h_eta.begin();
      hitr !=h_eta.end(); hitr++) {
    int eta = hitr->first;
    if(HBHE && fabs(eta)>29) continue;
    if(!HBHE && fabs(eta) <=29) continue;
    h->SetBinContent(40+eta+1,hitr->second.first);
    h->SetBinError(40+eta+1,std::sqrt(hitr->second.second));
  }
  h->Write();
  f.Close();
}


void RecJet::eta_distribution(std::string fname, bool var) {
  std::map<unsigned int,RecJet::MyInfo> m_ = LoopMap();
  char name[50], title[50];
  TFile f(fname.c_str(), "recreate");
  if (m_.size() == 0) return;
  std::map<unsigned int, TH1D*> h;
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  for(int idet=1; idet<=4; idet++){
    for(int depth=1; depth<=3; depth++){
      unsigned int detId = (((idet&0x7)<<5)|(depth&0x1f));
      if (var) {
	sprintf(name,"%s%d_eta",det[idet-1].c_str(),depth);
	sprintf(title,"Variance distribution w.r.t.eta for %s depth %d", det[idet-1].c_str(),depth);
      } else {
	sprintf(name,"%s%d_eta",det[idet-1].c_str(),depth);
        sprintf(title,"Energy distribution w.r.t.eta for %s depth %d", det[idet-1].c_str(),depth);
      }
      TH1D* hist = new TH1D(name,title,82, -41., 41.);
      if (var) {
	hist->GetYaxis()->SetTitle("Variance in Gev^2"); hist->GetYaxis()->CenterTitle();
      } else {
	hist->GetYaxis()->SetTitle("Mean Energy in Gev"); hist->GetYaxis()->CenterTitle();
      }
      hist->GetXaxis()->SetTitle("i#eta"); hist->GetXaxis()->CenterTitle();
      h[detId] = hist;
    }
  }

  std::map<unsigned int, std::pair<double,double> > h_eta;
  for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.begin();
       mitr !=m_.end(); mitr++) {
    int subd = ((mitr->first)>>20)&0x7;
    int eta = (mitr->first)&0x2000 ? ((mitr->first)>>7)&0x3f : -(((mitr->first)>>7)&0x3f);  
    int depth = ((mitr->first)>>14)&0x1f;
    unsigned int detId = (((subd&0x7)<<12) | ((depth&0x1f)<<7) | ((eta>0) ? (0x40|eta) : -eta));
    std::map<unsigned int,std::pair<double,double> >::iterator hitr = h_eta.find(detId);
    if (hitr == h_eta.end()) {
      double energy=0.;
      double error =0.;
      h_eta[detId] = std::pair<double,double>(energy,error);
      hitr          = h_eta.find(detId);
      }
    std::pair<double,double> mean, variance;
    MeanVariance(mitr,mean,variance);
    std::pair<double,double> nsub;
    if (var){
      nsub = SubtractNoise(mitr->first,variance.first,variance.second,true);
    } else {
      nsub = SubtractNoise(mitr->first,mean.first,mean.second,false);
    }
    hitr->second.first +=nsub.first;
    hitr->second.second +=pow(nsub.second,2);
  }
  for(std::map<unsigned int,std::pair<double,double> >::iterator hitr = h_eta.begin();
      hitr !=h_eta.end(); hitr++) {
    unsigned int detId = (((hitr->first)>>7)&0xff);
    int eta = (((hitr->first)&0x40) ? ((hitr->first)&0x3f) : -((hitr->first)&0x3f));
    for(std::map<unsigned int,TH1D* >::iterator itr = h.begin();
	itr !=h.end(); itr++){
      if(itr->first != detId)continue;
      itr->second->SetBinContent(40+eta+1,hitr->second.first);
      itr->second->SetBinError(40+eta+1,std::sqrt(hitr->second.second));
    }
  }
  for(std::map<unsigned int,TH1D* >::iterator itr = h.begin();
      itr !=h.end(); itr++) if(itr->second->GetEntries() !=0) itr->second->Write();
  f.Close();
}

void RecJet::det_distribution(std::string fname) {
  TFile f(fname.c_str(),"recreate");
  std::map<unsigned int,RecJet::MyInfo> m_ = LoopMap();
  if (m_.size() == 0) return;
  TH1D* h[4];
  std::string det[4] = {"HB", "HE","HO", "HF"};
  char        name[700], title[700];
  for (int idet=1; idet<=4; idet++) {
    sprintf(name, "%s", det[idet-1].c_str());
    sprintf (title, "Noise distribution for %s", det[idet-1].c_str());
    h[idet-1] = new TH1D(name,title,32,-4., 4.);
  }
  for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.begin();
       mitr !=m_.end(); mitr++) {
    int subd = ((mitr->first)>>20)&0x7;
    std::pair<double,double> mean, variance;
    MeanVariance(mitr,mean,variance);
    h[subd-1]->Fill(mean.first);
  }
  for (int i=0; i<4; i++) h[i]->Write();
}

void RecJet::Fit(std::string rootfile, std::string textfile) {
  TFile filex(rootfile.c_str());
  ofstream myfile;
  myfile.open(textfile.c_str());
  myfile << std::setw(25) << "Histo" << std::setw(8) << "Entries"
	 << std::setw(10) << "Mean" << std::setw(8) << "RMS" << std::setw(20)
	 << "Const_fit/Error" << std::setw(20) << "Mean_fit/Error"
	 << std::setw(20) << "Sigma_fit/Error" << std::setw(10)
         << "Chi2/ndf" << "\n" <<std::endl;
  std::string det[3] = {"HB", "HE", "HF"};
  for(int idet=0; idet<=2; idet++){
    for(int ietax=-41; ietax<=41; ietax++){
      for(int iphix=1; iphix<=72; iphix++){
        for(int dep=1; dep<=3; dep++){
	  std::ostringstream s;
          s << det[idet].c_str() << "eta" << ietax << "phi" << iphix << "dep" << dep;
	  std::string d = s.str();
          TH1D* h; filex.GetObject(d.c_str(),h);
          if(h && h->GetEntries()) {
            myfile << std::setw(25)  << h->GetTitle() << std::setw(8) 
		   << h->GetEntries() << std::setw(10) << std::setprecision(4) 
		   << h->GetMean() << std::setw(8) << std::setprecision(4) 
		   << h->GetRMS();
            TF1 *f1 = new TF1("f1","[0]*TMath::Exp((-0.5/pow([2],2))*pow((x[0] -[1]),2))",-3,3);
            f1->SetParameters(500,h->GetMean(),h->GetRMS());
            f1->SetParNames ("Constant","Mean_value","Sigma");
            h->Fit("f1","R");
            double par[3];
            f1->GetParameters(&par[0]);
            myfile << std::setw(10) << std::setprecision(4) << par[0]  
		   << std::setw(9) << std::setprecision(4) << f1->GetParError(0)
                   << std::setw(10) << std::setprecision(4) << par[1]
		   << std::setw(9) << std::setprecision(4) << f1->GetParError(1)
                   << std::setw(10) << std::setprecision(4) << par[2]
		   << std::setw(9) << std::setprecision(4) <<f1->GetParError(2)
                   <<std::setw(10) << f1->GetChisquare()/f1->GetNDF() << endl;
          }
        }
      }
    }
  }
}


void RecJet::Disturb(int subd) {
  std::cout << "Factor of multiplication: ";
  std::cin >> factor_;
  dets_.clear();
  std::cout << "Enter detid for subd=" << subd << std::endl; 
  int detId[3];
  std::string detid[3] = {"depth", "ieta", "iphi"};
  while (1) {
    bool ok(true);
    for (int i=0; i<3; i++) {
      std::cout << "Enter the value of " << detid[i].c_str() << std::endl;
      std::cin >> detId[i];
      if (i != 1 && detId[i] <= 0) {
	ok = false; break;
      }
    }
    if (ok) {
      unsigned int detId1 = ((subd<<20) | ((detId[0]&0x1f)<<14) | 
			     ((detId[1]>0) ? (0x2000|(detId[1]<<7)) :
			      ((-detId[1])<<7)) | (detId[2]&0x7f));
      dets_.push_back(detId1);
    } else {
      break;
    }
  }
}

std::map<unsigned int,RecJet::MyInfo> RecJet::MultiplyEnergy(std::map<unsigned int, RecJet::MyInfo>& m_, int corrfac) {

  if (corrfac > 0) {
    for (std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.begin();
	 mitr !=m_.end(); mitr++) {
      std::map<unsigned int,RecJet::CFactors>::iterator itr = corrFactor_.find(mitr->first);
      double fac = ((itr == corrFactor_.end()) ? 1 : 
		    ((corrfac == 1) ? itr->second.cfac1 : itr->second.cfac2));
      ChangeMoments(mitr,fac);
    }
  } else {
    for (unsigned k=0; k<dets_.size(); ++k) {
      std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.find(dets_[k]);
      if (mitr != m_.end()) ChangeMoments(mitr,factor_);
    }
  }
  return m_;
}  

void RecJet::ChangeMoments(std::map<unsigned int,RecJet::MyInfo>::iterator& mitr, double fac) {

  mitr->second.Mom1 *= fac;
  mitr->second.Mom2 *= (fac*fac);
  mitr->second.Mom3 *= (fac*fac*fac);
  mitr->second.Mom4 *= (fac*fac*fac*fac);
}
         
std::pair<double,double> RecJet::SubtractNoise(unsigned int detId, double mean,
					       double error, bool ifVar) {

  std::map<unsigned int,RecJet::MyInfo>::iterator mitr = mNoise_.find(detId);
  if (mitr != mNoise_.end()) {
    std::cout << "//////////////////enter" << std::endl;
    double m2(0), e2(0);
    if (ifVar) {
      double mean1    = mitr->second.Mom1 / mitr->second.Mom0;
      double mean2    = mitr->second.Mom2 / mitr->second.Mom0;
      double mean3    = mitr->second.Mom3 / mitr->second.Mom0;
      double mean4    = mitr->second.Mom4 / mitr->second.Mom0;
      m2 = (mean2 - mean1*mean1);
      e2 = ((mean4+8*mean1*mean1*mean2-4*mean1*mean3-4*mean1*mean1*mean1*mean1-
	     mean2*mean2)/mitr->second.Mom0);
    } else {
      m2 = mitr->second.Mom1 / mitr->second.Mom0 ;
      e2 = ((mitr->second.Mom2/mitr->second.Mom0)-m2*m2)/mitr->second.Mom0;
    }
    mean -= m2;
    error = std::sqrt(error*error+e2);
  }
  return std::pair<double,double>(mean,error);
}

std::map<unsigned int,RecJet::MyInfo> RecJet::LoadNoise() {

  std::map<unsigned int,RecJet::MyInfo> m_;
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  for (int subd=0; subd<4; ++subd) {
    char fname[80];
    sprintf (fname,"%s_Noise.txt",det[subd].c_str());
    ifstream myfile(fname);
    if (!myfile.is_open()) {
      std::cerr << "** ERROR: Can't open '" << fname << "' for the noise file" 
		<< std::endl;
    } else {
      unsigned int nrec(0), ndets(0);
      while(1) {
	unsigned int detId;
	double       mom0, mom1, mom2, mom3, mom4;
	myfile >> detId >> mom0 >> mom1 >> mom2 >> mom3 >> mom4;
	if (!myfile.good()) break;
	std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.find(detId);
	if (mitr == m_.end()) {
	  RecJet::MyInfo info;
	  m_[detId] = info ;
	  mitr = m_.find(detId) ;
	  ndets++;
	}
	mitr->second.Mom0 += mom0_MB;
	mitr->second.Mom1 += mom1_MB;
	mitr->second.Mom2 += mom2_MB;
	mitr->second.Mom3 += mom3_MB;
	mitr->second.Mom4 += mom4_MB;
	nrec++;
      }
      myfile.close();
      std::cout << "Reads " << nrec << " records for " << ndets << " elements"
		<< " in file " << fname << std::endl;
    }
  }
  return m_;
}

void RecJet::MeanVariance(std::map<unsigned int,RecJet::MyInfo>::iterator &mitr,
			  std::pair<double,double>& mean,
			  std::pair<double,double>& variance) {
  mean.first     = mitr->second.Mom1 / mitr->second.Mom0;
  double mean2   = mitr->second.Mom2 / mitr->second.Mom0;
  double mean3   = mitr->second.Mom3 / mitr->second.Mom0;
  double mean4   = mitr->second.Mom4 / mitr->second.Mom0;
  variance.first = (mean2 - mean.first*mean.first);
  mean.second    = std::sqrt(variance.first / mitr->second.Mom0);
  variance.second= std::sqrt((mean4+8*mean.first*mean.first*mean2-
			      4*mean.first*mean3-mean2*mean2-
			      4*mean.first*mean.first*mean.first*mean.first)/
			     mitr->second.Mom0);
}

/* Calculation of correction factor following staggered geometry which is not requird*/


 std::vector<std::pair<double,double> > RecJet::Staggered_CorrFactor(int subd, std::map<unsigned int,RecJet::Hists>::iterator &hitr, bool varmethod) {
  std::vector<std::pair<double,double> > corrfac;
  double       mean1(0.), mean2(0.), error1(0.), error2(0.);
  unsigned int i_in = (subd == 0) ? 0 : 1;
  unsigned int i_end = (subd == 0) ? 71 : 72 ; 
  if (varmethod) {
    for (unsigned int i=i_in; i<i_end; i=i+4) {
      mean1 += (i_in == 0) ? hitr->second.h2->GetBinContent(1)   + hitr->second.h2->GetBinContent(72) :
		             hitr->second.h2->GetBinContent(i)   + hitr->second.h2->GetBinContent(i+1) ; 
      mean2 +=               hitr->second.h2->GetBinContent(i+2) + hitr->second.h2->GetBinContent(i+3) ;
     
      error1 += (i_in == 0) ? pow((hitr->second.h2->GetBinError(1)), 2) + pow((hitr->second.h2->GetBinError(72)),2) : 
   			      pow((hitr->second.h2->GetBinError(i)), 2) + pow((hitr->second.h2->GetBinError(i+1)),2) ;
      error2 +=               pow((hitr->second.h2->GetBinError(i+2)),2)+ pow((hitr->second.h2->GetBinError(i+3)),2) ;
    }
    mean1  /= (hitr->second.h2->GetEntries()/2);
    mean2  /= (hitr->second.h2->GetEntries()/2);
    error1 /= (hitr->second.h2->GetEntries()/2);
    error2 /= (hitr->second.h2->GetEntries()/2);
  } else {
    for (unsigned int i=i_in; i<i_end; i=i+4) {
        mean1 += (i_in == 0) ? hitr->second.h1->GetBinContent(1)   + hitr->second.h1->GetBinContent(72) :
			       hitr->second.h1->GetBinContent(i)   + hitr->second.h1->GetBinContent(i+1) ; 
        mean2 +=               hitr->second.h1->GetBinContent(i+2) + hitr->second.h1->GetBinContent(i+3) ;
       
       error1 += (i_in == 0) ? pow((hitr->second.h1->GetBinError(1)), 2) + pow((hitr->second.h1->GetBinError(72)),2) : 
   			       pow((hitr->second.h1->GetBinError(i)), 2) + pow((hitr->second.h1->GetBinError(i+1)),2) ;
       error2 +=               pow((hitr->second.h1->GetBinError(i+2)),2)+ pow((hitr->second.h1->GetBinError(i+3)),2) ;
       
    }
    mean1  /= (hitr->second.h1->GetEntries()/2);
    mean2  /= (hitr->second.h1->GetEntries()/2);
    error1 /= (hitr->second.h1->GetEntries()/2);
    error2 /= (hitr->second.h1->GetEntries()/2);
  }
  
  double n     = (varmethod) ? hitr->second.h2->GetEntries() : hitr->second.h1->GetEntries();
  for (int i=1; i<=hitr->second.h1->GetNbinsX(); ++i) {
    double  mean =    (subd == 0) ? (((i-1)%4 == 0) || (i%4 == 0) ? mean1 : mean2) :
		      (subd == 1) ? (((i+1)%4 == 0) || (i%4 == 0) ? mean2 : mean1) : (mean1 + mean2)/2 ;

   double error =     (subd == 0) ? (((i-1)%4 == 0) || (i%4 == 0) ? error1 : error2) : 
		      (subd == 1) ? (((i+1)%4 == 0) || (i%4 == 0) ? error2 : error1) : (error2 + error1)/2;
   if(varmethod) {
     hitr->second.h8->SetBinContent(i,mean);
     hitr->second.h8->SetBinError(i,error);
     //     std::cout << error << std::endl;
   } else {
     hitr->second.h9->SetBinContent(i,mean);
     hitr->second.h9->SetBinError(i,error);
     std::cout << error << std::endl;
   }

   double vcont = (varmethod) ? hitr->second.h2->GetBinContent(i) : hitr->second.h1->GetBinContent(i);
   double econt = (varmethod) ? hitr->second.h2->GetBinError(i)   : hitr->second.h1->GetBinError(i);
   double vcfac = (varmethod) ? ((vcont > 0) ? (std::sqrt(mean/vcont)) : 1) : ((vcont > 0) ? (mean/vcont) : 1);
   double ecfac = (varmethod) ? ((vcont>0 && mean>0) ? (0.5*std::sqrt(((2*error)/(vcont*mean*n)) + (econt*econt)/(vcont*vcont) *((mean/vcont) - (4.0/n)))) : 0) : ((vcont>0 && mean>0) ? (std::sqrt(((error*2.)/(n*vcont*vcont)) + (econt/vcont)*(econt/vcont)*((mean/vcont)*(mean/vcont)-(4.0/n)*(mean/vcont)))) : 0);
    corrfac.push_back(std::pair<double,double>(vcfac,ecfac));
  }
  return corrfac;
}  


/* Calculation of correction factor where staggered geometry is not followed. This should be followed not previous one*/ 

std::vector<std::pair<double,double> > RecJet::CorrFactor(int , std::map<unsigned int,RecJet::Hists>::iterator &hitr, bool varmethod) {
  std::vector<std::pair<double,double> > corrfac;
  double       mean(0.), error(0.);
  if (varmethod) {
    for (unsigned int i=1; i<=72; i++) {
      mean  +=  hitr->second.h2->GetBinContent(i);
      error +=  pow((hitr->second.h2->GetBinError(i)), 2);
    }
    mean  /= hitr->second.h2->GetEntries();
    error /= hitr->second.h2->GetEntries();
  } else {
    for (unsigned int i=1; i<=72; i++) {
      mean  += hitr->second.h1->GetBinContent(i);
      error += pow((hitr->second.h1->GetBinError(i)), 2) ;
    }
    mean  /= hitr->second.h1->GetEntries();
    error /= hitr->second.h1->GetEntries();
  }

  for (int i=1; i<=hitr->second.h1->GetNbinsX(); ++i) {
    if(varmethod) {
      hitr->second.h8->SetBinContent(i,mean);
      hitr->second.h8->SetBinError(i,std::sqrt(error));
    } else {
      hitr->second.h9->SetBinContent(i,mean);
      hitr->second.h9->SetBinError(i,std::sqrt(error));
    }
    double n     = (varmethod) ? hitr->second.h2->GetEntries() : hitr->second.h1->GetEntries();
    double vcont = (varmethod) ? hitr->second.h2->GetBinContent(i) : hitr->second.h1->GetBinContent(i);
    double econt = (varmethod) ? hitr->second.h2->GetBinError(i)   : hitr->second.h1->GetBinError(i);
    double vcfac = (varmethod) ? ((vcont > 0) ? (std::sqrt(mean/vcont)) : 1) : ((vcont > 0) ? (mean/vcont) : 1);
    double ecfac = (varmethod) ? ((vcont>0 && mean>0) ? (0.5*std::sqrt(((error)/(vcont*mean*n)) + (econt*econt)/(vcont*vcont) *((mean/vcont)-(2.0/n)))) : 0) : ((vcont>0 && mean>0) ? (std::sqrt(((error)/(n*vcont*vcont)) + (econt/vcont)*(econt/vcont)*((mean/vcont)*(mean/vcont)-(2.0/n)*(mean/vcont)))) : 0);
    corrfac.push_back(std::pair<double,double>(vcfac,ecfac));
  }
  return corrfac;
}

void RecJet::StoreCorrFactor(const std::map<unsigned int,RecJet::Hists>& hh_,
			     std::map<unsigned int,RecJet::CFactors>& cfacs,
			     bool clear) {

  if (clear) cfacs.clear();
  for (std::map<unsigned int,RecJet::Hists>::const_iterator hitr = hh_.begin();
       hitr != hh_.end(); ++hitr) {
    unsigned int detId2  = (hitr->first);
    if (hitr->second.h3 != 0) {
      for (int i=1; i<=hitr->second.h3->GetNbinsX(); ++i) {
	if (hitr->second.h3->GetBinContent(i) > 0) {
	  unsigned int detId = (detId2 | i);
	  std::map<unsigned int,RecJet::CFactors>::iterator itr = cfacs.find(detId);
	  if (itr == cfacs.end()) {
	    RecJet::CFactors fac;
	    cfacs[detId] = fac;
	    itr          = cfacs.find(detId);
	  }
	  itr->second.cfac1 = (hitr->second.h3->GetBinContent(i));
	  itr->second.efac1 = (hitr->second.h3->GetBinError(i));
	}
      }
    }
    if (hitr->second.h4 != 0) {
      for (int i=1; i<=hitr->second.h4->GetNbinsX(); ++i) {
	if (hitr->second.h4->GetBinContent(i) > 0) {
	  unsigned int detId = (detId2 | i);
	  std::map<unsigned int,RecJet::CFactors>::iterator itr = cfacs.find(detId);
	  if (itr == cfacs.end()) {
	    RecJet::CFactors fac;
	    cfacs[detId] = fac;
	    itr          = cfacs.find(detId);
	  }
	  itr->second.cfac2 = (hitr->second.h4->GetBinContent(i));
	  itr->second.efac2 = (hitr->second.h4->GetBinError(i));
	}
      }
    }
  }
}

std::pair<double,double> RecJet::MeanDeviation(int sdet, std::map<unsigned int,RecJet::CFactors>& cfacs) {
  double dev1(0), dev2(0);
  int    kount(0);
  for (std::map<unsigned int,RecJet::CFactors>::iterator itr=cfacs.begin();
       itr != cfacs.end(); ++itr) {
    int sdet0 = ((itr->first)>>20)&0x7;
    if (sdet0 == sdet) {
      kount++;
      dev1 +=  (((itr->second.cfac1) > 1.0) ? (1.0-(1.0/(itr->second.cfac1))) : (1.0-(itr->second.cfac1)));
      dev2 +=  (((itr->second.cfac2) > 1.0) ? (1.0-(1.0/(itr->second.cfac2))) : (1.0-(itr->second.cfac2)));
    }
  }
  if (kount > 0) {
    dev1 /= kount;
    dev2 /= kount;
  }
  std::cout << kount << " dev " << dev1 << ":" << dev2 << std::endl;
  return std::pair<double,double>(dev1,dev2);
}

void RecJet::WriteCorrFactor(std::string& outFile, std::string& inFile, bool var) {

  // Read (if exists) correction factors from an old file
  std::map<unsigned int, std::pair<double,double> > corrFactor;
  ifstream myfile;
  myfile.open(inFile.c_str());
  if (!myfile.is_open()) {
    std::cout << "** ERROR: Can't open file '" << inFile << "' for i/p"
	      << std::endl;
  } else {
    while(1) {
      unsigned int detId;
      double       cfac, efac;
      myfile >> detId >> cfac >> efac;
      if (!myfile.good()) break;
      std::map<unsigned int,std::pair<double,double> >::iterator itr = corrFactor.find(detId);
      if (itr == corrFactor.end()) {
	corrFactor[detId].first = cfac;
	corrFactor[detId].second= efac;
      } else {
	itr->second.first  = cfac;
	itr->second.second = efac;
      }
    }
    myfile.close();
  }

  // See if it is the same as the ones calculated
  double diff(0);
  int    kount(0);
  const  unsigned int det(4);
  for (std::map<unsigned int,RecJet::CFactors>::iterator itr = corrFactor_.begin();
       itr != corrFactor_.end(); ++itr) {
    unsigned int sdet    = ((itr->first)>>20)&0x7;
    unsigned int detId   = ((det&0xf)<<28)|((sdet&0x7)<<25);
    detId               |= ((itr->first)&0xfffff);
    std::map<unsigned int,std::pair<double,double> >::iterator citr = corrFactor.find(detId);
    double cfac = (var) ? (itr->second.cfac2) : (itr->second.cfac1);
//    double efac = (var) ? (itr->second.efac2) : (itr->second.efac1);
    if (citr != corrFactor.end() && cfac != 0) {
      kount++;
      //std::cout << cfac << "\t" << citr->second << "\t" << kount << std::endl;
      diff += fabs(cfac - citr->second.first);
    }
  }
  bool same = (kount > 0) ? (diff/kount < 0.0001) : false;
  std::cout << "Results from comparison: " << diff << " in " << kount << " --> "
	    << same << std::endl;

  // Now write it out
  ofstream myfile1;
  myfile1.open(outFile.c_str());
  if (!myfile1.is_open()) {
    std::cout << "** ERROR: Can't open '" << outFile << std::endl;
  } else {
    myfile1 << std::setprecision(4) << std::setw(10) << "detId" 
	    << std::setw(10) << "sub_det" << std::setw(10) << "ieta" 
	    << std::setw(10) << "iphi" << std::setw(10) << "depth" 
	    << std::setw(10) << "corrfact" << std::setw(15) << "error" 
	    << std::endl;
    for (std::map<unsigned int,RecJet::CFactors>::iterator itr = corrFactor_.begin();
	 itr != corrFactor_.end(); ++itr) {
      double cfac = (var) ? (itr->second.cfac2) : (itr->second.cfac1);
      double efac = (var) ? (itr->second.efac2) : (itr->second.efac1);
      if (cfac != 0) {
	unsigned int sdet    = ((itr->first)>>20)&0x7;
	unsigned int detId   = ((det&0xf)<<28)|((sdet&0x7)<<25);
	detId               |= ((itr->first)&0xfffff);
	std::map<unsigned int,std::pair<double,double> >::iterator citr = corrFactor.find(detId);
	if (citr != corrFactor.end() && (!same)) {
	  cfac *= (citr->second.first);
	  efac *= (citr->second.second);
	}
	int keta  = ((detId&0x2000) ? ((detId>>7)&0x3f) : -((detId>>7)&0x3f));
        int depthx= ((detId>>14)&0x1f);
        int kphi  = (detId&0x7f);
        myfile1 << std::setw(10) << std::hex << detId 
		<< std::setw(10) << std::dec << sdet 
		<< std::setw(10) << keta
                << std::setw(10) << kphi  
		<< std::setw(10) << depthx
		<< std::setw(10) << cfac 
		<< std::setw(15) << efac
	        << std::endl;
      }
    }
    myfile1.close();
  }
}

void RecJet::Error_study(std::string fname, std::string det, int depth){

  int detector = (det == "HB") ? 1 : ( (det == "HE") ? 2 : 4);
  char name[100], title[100];   
  
  sprintf(name,"%s_%d_error", det.c_str(), depth);
  sprintf(title,"%s_depth%d_error", det.c_str(), depth);
  TH1F* h_err = new TH1F(name,title,100, 0., 0.005); 
  h_err->GetXaxis()->SetTitle("Error");  h_err->GetXaxis()->CenterTitle();
  
  sprintf(name,"%s_%d_corr", det.c_str(), depth);
  sprintf(title,"%s_depth%d_corr", det.c_str(), depth);
  TH1F* h_corr = new TH1F(name,title,100, 0., 2);
  h_corr->GetXaxis()->SetTitle("Corr_Factor"); h_corr->GetXaxis()->CenterTitle();
  
  sprintf (name, "%s_depth%d", det.c_str(), depth);
  TCanvas* c  = new TCanvas(name, name, 500, 500);
  c->SetRightMargin(0.10);
  c->SetTopMargin(0.10);
  gStyle->SetOptStat(111111);
  c->Divide(2,1);
  c->cd(1);
  
  ifstream myfile;
  sprintf(name, "%s", fname.c_str());
  std::cout << name << std::endl;
  myfile.open(name);
  if (!myfile.is_open()) {
    std::cout << "** ERROR: Can't open file '" << std::endl;
  } else {
    while(1) {
      unsigned int detId;
      int subdet, ieta, iphi, idepth;
      double       cfac, efac;
      myfile >> std::hex >> detId >> std::dec >> subdet >> ieta >> iphi >> idepth >> cfac >> efac;
      //      std::cout << std::hex << detId << std::dec << subdet << ieta << iphi << idepth << cfac << efac << std::endl;
      if (!myfile.good()) break;
      if(subdet != detector || idepth != depth) continue;
      h_err->Fill(efac);
      h_corr->Fill(cfac);
    }
    myfile.close();
  }
  h_corr->Draw();
  c->cd(2);
  h_err->Draw();
}

void RecJet::LoopMapTrig() {
  //   In a ROOT session, you can do:
  //      Root > .L RecJet.C
  //      Root > RecJet t
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

  mTrig_.clear();
  if (fChain != 0) {
    Long64_t nentries = fChain->GetEntriesFast();
    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      unsigned int detId = ((mysubd<<20) | ((depth&0x1f)<<14) | 
			    ((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) | 
			    (iphi&0x7f));
      std::pair<unsigned int,int>key(detId,trigbit);
      std::map<std::pair<unsigned int,int>,RecJet::MyInfo>::iterator mitr = mTrig_.find(key);
      if (mitr == mTrig_.end()) {
	RecJet::MyInfo info;
	mTrig_[key] = info ;
	mitr = mTrig_.find(key) ;
      }
      mitr->second.Mom0 += mom0_MB;
      mitr->second.Mom1 += mom1_MB;
      mitr->second.Mom2 += mom2_MB;
      mitr->second.Mom3 += mom3_MB;
      mitr->second.Mom4 += mom4_MB;
    }
    loadTrig_ = true;
  }  
}

void RecJet::LoopTrig(int sdet, int eta, int phi, int dep, std::string indx) {

  std::string det[4] = {"HB", "HE", "HO", "HF"};
  int subd = ((sdet>=1 && sdet<=4) ? sdet : 1);

  if (!loadTrig_) LoopMapTrig();
  unsigned int detId = ((subd<<20) | ((dep&0x1f)<<14) | 
			((eta>0)?(0x2000|(eta<<7)):((-eta)<<7)) | (phi&0x7f));
  char fname[80],name[80],title[100];
  sprintf (fname, "dTrigbit%s_%s%d%d%d.root",indx.c_str(),det[subd-1].c_str(),eta,phi,dep);
  TFile f(fname, "recreate");
  sprintf(name, "mom0%s%d%d%d",det[subd-1].c_str(),eta,phi,dep);
  sprintf(title, "Entries for %s i#eta=%d i#phi=%d depth=%d",det[subd-1].c_str(),eta,phi,dep);
  TH1I* h1 = new TH1I(name, title, 128, 0, 128);
  sprintf(name, "edm0%s%d%d%d",det[subd-1].c_str(),eta,phi,dep);
  sprintf(title, "Error/Mean for %s i#eta=%d i#phi=%d depth=%d",det[subd-1].c_str(),eta,phi,dep);
  TH1D* h2 = new TH1D(name, title, 128, 0, 128);
  std::string bitn[113] = {"ZeroBias", "AlwaysTrue",
			   "Mu16er_TauJet40erORCenJet72er", "Mu12_EG10",
			   "Mu5_EG15", "DoubleTauJet40er", 
			   "IsoEG20er_TauJet20er_NotWdEta0", 
			   "Mu16er_TauJet20er", "QuadJetC84", 
			   "DoubleJetC72", "DoubleJetC120",
			   "SingleJet52", "SingleJet68", "SingleJet92",
			   "SingleJet128", "SingleJet176", "SingleJet200",
			   "SingleJet36", "DoubleTauJet36er", 
			   "DoubleTauJet44er", "DoubleMu0", 
			   "SingleMu30er", "Mu3_JetC92_WdEtaPhi2",
			   "Mu3_JetC16_WdEtaPhi2", "Mu3_JetC52_WdEtaPhi2",
			   "EG25er_HTT125", "SingleEG35er", "SingleIsoEG25er",
			   "SingleIsoEG25", "SingleIsoEG28er",
			   "SingleIsoEG30er", "SingleEG10", "SingleIsoEG22er",
			   "SingleJet240", "DoubleJetC52", "DoubleJetC84",
			   "SingleMu14_Eta2p1", "DoubleJetC112",
			   "DoubleMu_10_Open", "DoubleMu_10_3p5", "QuadJetC40",
			   "SingleEG5", "SingleEG25", "SingleEG40",
			   "SingleIsoEG18", "SingleIsoEG20er", "SingleEG20",
			   "SingleEG30", "SingleEG35", "SingleMuOpen",
			   "SingleMu16", "SingleMu5", "SingleMu20er",
			   "SingleMu12", "SingleMu20", "SingleMu25er",
			   "SingleMu25", "SingleMu30", "ETM30", "ETM50",     
			   "ETM70", "ETM100", "HTT125", "HTT150", "HTT175",
			   "HTT200", "Mu20_EG10", "Mu5_EG20", "Mu5_IsoEG18",
			   "Mu6_DoubleEG", "SingleJetC32_NotBptxOR",
			   "ETM40", "HTT250", "Mu20_EG8", "Mu6_HTT150",
			   "Mu10er_ETM50", "Mu14er_ETM30", "DoubleMu7_EG7",
			   "SingleMu16_Eta2p1", "Mu8_HTT125", "Mu4_EG18",
			   "SingleMu6_NotBptxOR", "DoubleMu6_EG6",
			   "Mu5_DoubleEG5", "DoubleEG6_HTT150", 
			   "QuadJetC36_TauJet52", "TripleMu0", "TripleMu_5_5_3",
			   "TripleEG_14_10_8", "DoubleEG_15_10", 
			   "DoubleEG_22_10", "DoubleEG_20_10_1LegIso", 
			   "Mu0er_ETM55", "DoubleJetC60_ETM60",
			   "DoubleJetC32_WdPhi7_HTT125",
			   "Jet32_DoubleMu_Open_10_MuMuNotWdPhi23_JetMuWdPhi1",
			   "Jet32_MuOpen_EG10_MuEGNotWdPhi3_JetMuWdPhi1",
			   "ETM60", "DoubleJetC100", "QuadJetC60",
			   "SingleJetC20_NotBptxOR", "DoubleJetC56_ETM60",
			   "DoubleMu0_Eta1p6_WdEta18", "ETM60_NotJet52WdPhi2",
			   "ETM70_NotJet52WdPhi2", "TripleEG5",
			   "TripleJet_92_76_64", "QuadMu0", "SingleMu18er",
			   "DoubleMu0_Eta1p6_WdEta18_OS", "DoubleMu_12_5",
			   "DoubleMu_10_0_WdEta18", "SingleMuBeamHalo"};
  int ibit[113] = {0,     1,   8,   9,  10,  11,  12,  13,  14,  15,  16,
		   17,   18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
		   28,   29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
		   39,   40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
		   50,   51,  52,  53,  54,  55,  56,  58,  60,  61,  62,
		   63,   64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
		   74,   75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
		   85,   86,  87,  88,  89,  90,  91,  92,  93,  97,  98,
		   100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
		   111, 112, 113, 114, 115, 116, 117, 119, 121, 122, 123,
		   124, 126, 127};
  for (unsigned int k=0; k<113; ++k) {
    if (k%5 == 0) {
      h1->GetXaxis()->SetBinLabel(ibit[k]+1, bitn[k].c_str());
      h2->GetXaxis()->SetBinLabel(ibit[k]+1, bitn[k].c_str());
    }
  }

  for (std::map<std::pair<unsigned int,int>,RecJet::MyInfo>::iterator mitr = mTrig_.begin();
       mitr !=mTrig_.end(); mitr++) {
    if (mitr->first.first == detId) {
      int bin     = mitr->first.second;
      double mean = mitr->second.Mom1/mitr->second.Mom0;
      double error= std::sqrt(((mitr->second.Mom2/mitr->second.Mom0)-mean*mean)/mitr->second.Mom0);
      std::pair<double,double> nsub = SubtractNoise(detId,mean,error,false);
      double edm = nsub.second/nsub.first;
      std::cout << "Before noise " << mean << " +- " << error << " after " << nsub.first << " +- " << nsub.second << " edm " << edm << std::endl;
      if (bin > 0 && bin <= 128) {
	h1->SetBinContent(bin+1,mitr->second.Mom0);
	h2->SetBinContent(bin+1,edm);
      }
    }
  }
  h1->Write(); 
  h2->Write();  
  f.Close();
}

std::map<unsigned int,RecJet::MyInfo> RecJet::BuildMap(int sdet, bool cfirst, double emin, double emax) {

  std::map<unsigned int,RecJet::MyInfo> m_;
  int subd = ((sdet>=1 && sdet<=4) ? (sdet) : 0);
  std::string det[4] = {"HB", "HE", "HO", "HF"};
  int etamin[4] = {-16, -29, -15, -41};
  int etamax[4] = { 16,  29,  15,  41};
  int depmax[4] = {  2,   3,   4,   2};
  for (int eta=etamin[subd]; eta<=etamax[subd]; ++eta) {
    for (int phi=1; phi<=72; ++phi) {
      for (int depthx=1; depthx<=depmax[subd]; ++depthx) {
	char name[20];
	sprintf (name, "%seta%dphi%ddep%d", det[subd].c_str(), eta, phi, depthx);
	TH1D* hist = (TH1D*)file->FindObjectAny(name);
	if (hist != 0) {
	  unsigned int detId = ((subd<<20) | ((depthx&0x1f)<<14) | 
				((eta>0)?(0x2000|(eta<<7)):((-eta)<<7)) | 
				(phi&0x7f));
	  double cfac = (cfirst) ? 1.0 : corrFactor_[detId].cfac1;
	  std::map<unsigned int,RecJet::MyInfo>::iterator mitr = m_.find(detId);
	  if (mitr == m_.end()) {
	    RecJet::MyInfo info;
	    m_[detId] = info ;
	    mitr = m_.find(detId) ;
	  }
	  for (int i=1; i<=hist->GetNbinsX(); ++i) {
	    double e = cfac*(hist->GetBinLowEdge(i)+0.5*hist->GetBinWidth(i));
	    if (e > emin && e < emax) {
	      double cont = hist->GetBinContent(i);
	      mitr->second.Mom0 += cont;
	      mitr->second.Mom1 += (cont*e);
	      mitr->second.Mom2 += (cont*e*e);
	      mitr->second.Mom3 += (cont*e*e*e);
	      mitr->second.Mom4 += (cont*e*e*e*e);
	    }
	  }
	}
      }
    }
  }
  return m_;
}
