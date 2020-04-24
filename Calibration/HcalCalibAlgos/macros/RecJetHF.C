//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue May  2 15:43:18 2017 by ROOT version 5.34/19
// from TTree RecJetHF/RecJetHF Tree
// found on file: HCALHF.root
//////////////////////////////////////////////////////////

#include <TCanvas.h>
#include <TChain.h>
#include <TDirectory.h>
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
#include "TLegend.h"

#include <iomanip>
#include <sstream>

class RecJetHF {
public:
  struct MyInfo {
    double Mom0_F1, Mom1_F1, Mom2_F1,Mom0_F2, Mom1_F2, Mom2_F2;

    MyInfo() {
      Mom0_F1 = Mom1_F1 = Mom2_F1 = Mom0_F2 = Mom1_F2 = Mom2_F2 =0.;
    }
  };

  struct Hists {
    TH1D *h1, *h2, *h3, *h4;
    Hists() {
      h1 = h2 =h3 =h4 =0;
    }
  };

  RecJetHF(std::string fname, bool ratio=false);
  virtual ~RecJetHF();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual std::map<unsigned int,RecJetHF::MyInfo> LoopMap();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  std::map<unsigned int, RecJetHF::Hists> MakeRatio(std::map<unsigned int,RecJetHF::MyInfo>& m_);

private :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           cells;
  Int_t           mysubd;
  Int_t           depth;
  Int_t           ieta;
  Int_t           iphi;
  Float_t         mom0_F1;
  Float_t         mom1_F1;
  Float_t         mom2_F1;
  Float_t         mom3_F1;
  Float_t         mom4_F1;
  Float_t         mom0_F2;
  Float_t         mom1_F2;
  Float_t         mom2_F2;
  Float_t         mom3_F2;
  Float_t         mom4_F2;
  Int_t           trigbit;
  Double_t        rnnumber;
  
  // List of branches
  TBranch        *b_cells;   //!
  TBranch        *b_mysubd;   //!
  TBranch        *b_depth;   //!
  TBranch        *b_ieta;   //!
  TBranch        *b_iphi;   //!
  TBranch        *b_mom0_F1;   //!
  TBranch        *b_mom1_F1;   //!
  TBranch        *b_mom2_F1;   //!
  TBranch        *b_mom3_F1;   //!
  TBranch        *b_mom4_F1;   //!
  TBranch        *b_mom0_F2;   //!
  TBranch        *b_mom1_F2;   //!
  TBranch        *b_mom2_F2;   //!
  TBranch        *b_mom3_F2;   //!
  TBranch        *b_mom4_F2;   //!
  TBranch        *b_trigbit;   //!
  TBranch        *b_rnnumber;   //!
  
  TFile          *file;
  bool            ratio_;

};

RecJetHF::RecJetHF(std::string fname, bool ratio) : fChain(0), ratio_(ratio) {

  file = new TFile(fname.c_str());
  TDirectory* directory = (TDirectory*)(file->FindObjectAny("recAnalyzerHF"));
  TTree *tree = (TTree*)(directory->FindObjectAny("RecJet"));
  Init(tree);
}

RecJetHF::~RecJetHF() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t RecJetHF::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t RecJetHF::LoadTree(Long64_t entry) {
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

void RecJetHF::Init(TTree *tree) {
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
  fChain->SetBranchAddress("mom0_F1", &mom0_F1, &b_mom0_F1);
  fChain->SetBranchAddress("mom1_F1", &mom1_F1, &b_mom1_F1);
  fChain->SetBranchAddress("mom2_F1", &mom2_F1, &b_mom2_F1);
  fChain->SetBranchAddress("mom3_F1", &mom3_F1, &b_mom3_F1);
  fChain->SetBranchAddress("mom4_F1", &mom4_F1, &b_mom4_F1);
  fChain->SetBranchAddress("mom0_F2", &mom0_F2, &b_mom0_F2);
  fChain->SetBranchAddress("mom1_F2", &mom1_F2, &b_mom1_F2);
  fChain->SetBranchAddress("mom2_F2", &mom2_F2, &b_mom2_F2);
  fChain->SetBranchAddress("mom3_F2", &mom3_F2, &b_mom3_F2);
  fChain->SetBranchAddress("mom4_F2", &mom4_F2, &b_mom4_F2);
  fChain->SetBranchAddress("trigbit", &trigbit, &b_trigbit);
  fChain->SetBranchAddress("rnnumber", &rnnumber, &b_rnnumber);
  Notify();
}

Bool_t RecJetHF::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void RecJetHF::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t RecJetHF::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

std::map<unsigned int,RecJetHF::MyInfo> RecJetHF::LoopMap() {
  //   In a ROOT session, you can do: 
  //      Root > .L RecJetHF.C
  //      Root > RecJetHF t
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
  
  std::map<unsigned int,RecJetHF::MyInfo> m_;
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
      std::map<unsigned int, RecJetHF::MyInfo>::iterator mitr = m_.find(detId1);
      if (mitr == m_.end()) {
	RecJetHF::MyInfo info;
        m_[detId1] = info ;
        mitr = m_.find(detId1) ;
      }
      mitr->second.Mom0_F1 += mom0_F1;
      mitr->second.Mom1_F1 += mom1_F1;
      mitr->second.Mom2_F1 += mom2_F1;
      mitr->second.Mom0_F2 += mom0_F2;
      mitr->second.Mom1_F2 += mom1_F2;
      mitr->second.Mom2_F2 += mom2_F2;
    }
  }
  return m_;
}

void RecJetHF::Loop() {

  std::map<unsigned int,RecJetHF::MyInfo> m_ = LoopMap();
  if (m_.size() == 0) return;

  char fname[80];
  
  sprintf (fname, "SingleNeutrino2017HF_Prereco.root");
  TFile f(fname, "recreate");
  std::map<unsigned int,RecJetHF::Hists> hh_ = MakeRatio(m_);
  
  for (std::map<unsigned int,RecJetHF::Hists>::const_iterator hitr = hh_.begin();
       hitr != hh_.end(); ++hitr) {
    if (hitr->second.h1  != 0) hitr->second.h1->Write();
    if (hitr->second.h2  != 0) hitr->second.h2->Write();
    //    if (hitr->second.h3  != 0) hitr->second.h3->Write();
    //if (hitr->second.h4  != 0) hitr->second.h4->Write();

  }
  f.Close();
}

std::map<unsigned int, RecJetHF::Hists>
RecJetHF::MakeRatio(std::map<unsigned int,RecJetHF::MyInfo>& m_) {

  char name[80], title[80];
  int keta(0), kphi(0);

  std::map<unsigned int, RecJetHF::Hists> hh_ ;
  for (std::map<unsigned int,RecJetHF::MyInfo>::iterator mitr = m_.begin();
       mitr !=m_.end(); mitr++) {
    unsigned int sdet0  = ((mitr->first)>>20)&0x7;

    if (sdet0 == 4) {
      unsigned int detId2  = (mitr->first)&0x7fff80 ;
      std::map<unsigned int,RecJetHF::Hists>::iterator hitr = hh_.find(detId2);
      if (hitr == hh_.end()) {
	RecJetHF::Hists hh;
        hh_[detId2] = hh;
        hitr = hh_.find(detId2);
        keta = (((mitr->first)&0x2000) ? (((mitr->first)>>7)&0x3f) :
		-(((mitr->first)>>7)&0x3f));
        int dept = (((mitr->first)>>14)&0x1f);
	sprintf (name,  "histE1%d %d", keta, dept);
	if (ratio_) 
	  sprintf (title, "<E1/(E1+E2)> (i#eta %d depth %d)", keta, dept);
	else
	  sprintf (title, "<E1>/(<E1>+<E2>) (i#eta %d depth %d)", keta, dept);
        hitr->second.h1 = new TH1D(name, title, 72,0.,72.);
        hitr->second.h1->GetXaxis()->SetTitle("i#phi");
        hitr->second.h1->GetXaxis()->CenterTitle();
	if (ratio_) 
	  hitr->second.h1->GetYaxis()->SetTitle("<E1/(E1+E2)>");
	else
	  hitr->second.h1->GetYaxis()->SetTitle("<E1>/(<E1>+<E2>)");
        hitr->second.h1->GetYaxis()->CenterTitle();
	sprintf (name,  "histE2%d %d", keta, dept);
	if (ratio_) 
	  sprintf (title, "<E2/(E1+<E2)> (i#eta %d depth %d)", keta, dept);
	else
	  sprintf (title, "<E2>/(<E1>+<E2>) (i#eta %d depth %d)", keta, dept);
	hitr->second.h2 = new TH1D(name, title, 72,0.,72.);
        hitr->second.h2->GetXaxis()->SetTitle("i#phi");
        hitr->second.h2->GetXaxis()->CenterTitle();
	if (ratio_) 
	  hitr->second.h2->GetYaxis()->SetTitle("<E2/(E1+E2)>");
	else
	  hitr->second.h2->GetYaxis()->SetTitle("<E2>/(<E1>+<E2>)");
        hitr->second.h2->GetYaxis()->CenterTitle();
      }
      kphi = ((mitr->first)&0x7f); 

      double E1 = mitr->second.Mom1_F1;
      double E2 = mitr->second.Mom1_F2;
      double mom0 = mitr->second.Mom0_F1;
      double mom2_E1 = mitr->second.Mom2_F1;
      double mom2_E2 = mitr->second.Mom2_F2;
      double err_E1 = std::sqrt((mom2_E1/mom0 - (E1*E1)/(mom0*mom0))/mom0);
      double err_E2 = std::sqrt((mom2_E2/mom0 - (E2*E2)/(mom0*mom0))/mom0);
      double val1   = ratio_ ? E1/mom0 : E1/(E1+E2);
      double val2   = ratio_ ? E2/mom0 : E2/(E1+E2);
      double err1   = ratio_ ? err_E1 :
	((E1+E2)* err_E1 - E1*(err_E1+err_E2)) / (pow((E1+E2),2)*mom0);
      double err2   = ratio_ ? err_E2 :
	((E1+E2)* err_E2 - E2*(err_E1+err_E2)) / (pow((E1+E2),2)*mom0);
      hitr->second.h1->SetBinContent(kphi,val1);
      hitr->second.h1->SetBinError(kphi,err1);
      hitr->second.h2->SetBinContent(kphi,val2);
      hitr->second.h2->SetBinError(kphi,err2);
    }
  }
  return hh_;
}
