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

class CalibTreeExtended {
public :

  CalibTreeExtended(const char* infile, const char* outfile,
		    bool debug=false);
  virtual ~CalibTreeExtended();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

private:
  void             BookHisto(const char* outfile);
  void             Close();

  TTree                     *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t                      fCurrent;//!current Tree number in a TChain

  // Fixed size dimensions of array or collections stored in the TTree if any.
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
  std::vector<double>       *t_mapP;
  std::vector<double>       *t_mapPt;
  std::vector<double>       *t_mapEta;
  std::vector<double>       *t_mapPhi;
  Double_t                   t_mindR1;
  Double_t                   t_mindR2;
  Double_t                   t_eMipDR;
  Double_t                   t_eHcal;
  Double_t                   t_eHcal10;
  Double_t                   t_eHcal30;
  Double_t                   t_hmaxNearP;
  Double_t                   t_emaxNearP;
  Double_t                   t_eAnnular;
  Double_t                   t_hAnnular;
  Double_t                   t_rhoh;
  Bool_t                     t_selectTk;
  Bool_t                     t_qltyFlag;
  Bool_t                     t_qltyMissFlag;
  Bool_t                     t_qltyPVFlag;
  Double_t                   t_gentrackP;
  Double_t                   t_gentrackE;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies;
  std::vector<bool>         *t_trgbits;
  std::vector<unsigned int> *t_DetIds1;
  std::vector<double>       *t_HitEnergies1;
  std::vector<unsigned int> *t_DetIds3;
  std::vector<double>       *t_HitEnergies3;
  std::vector<unsigned int> *t_DetIdEC;
  std::vector<double>       *t_HitEnergyEC;
  std::vector<double>       *t_HitDistEC;
  std::vector<unsigned int> *t_DetIdHC;
  std::vector<double>       *t_HitEnergyHC;
  std::vector<double>       *t_HitDistHC;
  
  // List of branches
  TBranch                   *b_t_Run;
  TBranch                   *b_t_Event;
  TBranch                   *b_t_DataType;
  TBranch                   *b_t_ieta;
  TBranch                   *b_t_iphi;
  TBranch                   *b_t_EventWeight;
  TBranch                   *b_t_nVtx;
  TBranch                   *b_t_nTrk;
  TBranch                   *b_t_goodPV;
  TBranch                   *b_t_l1pt;
  TBranch                   *b_t_l1eta;
  TBranch                   *b_t_l1phi;
  TBranch                   *b_t_l3pt;
  TBranch                   *b_t_l3eta;
  TBranch                   *b_t_l3phi;
  TBranch                   *b_t_p;
  TBranch                   *b_t_pt;
  TBranch                   *b_t_phi;
  TBranch                   *b_t_mapP;
  TBranch                   *b_t_mapPt;
  TBranch                   *b_t_mapEta;
  TBranch                   *b_t_mapPhi;
  TBranch                   *b_t_mindR1;
  TBranch                   *b_t_mindR2;
  TBranch                   *b_t_eMipDR;
  TBranch                   *b_t_eHcal;
  TBranch                   *b_t_eHcal10;
  TBranch                   *b_t_eHcal30;
  TBranch                   *b_t_hmaxNearP;
  TBranch                   *b_t_emaxNearP;
  TBranch                   *b_t_eAnnular;
  TBranch                   *b_t_hAnnular;
  TBranch                   *b_t_rhoh;
  TBranch                   *b_t_selectTk;
  TBranch                   *b_t_qltyFlag;
  TBranch                   *b_t_qltyMissFlag;
  TBranch                   *b_t_qltyPVFlag;
  TBranch                   *b_t_gentrackP;
  TBranch                   *b_t_gentrackE;
  TBranch                   *b_t_DetIds;
  TBranch                   *b_t_HitEnergies;
  TBranch                   *b_t_trgbits;
  TBranch                   *b_t_DetIds1;
  TBranch                   *b_t_HitEnergies1;
  TBranch                   *b_t_DetIds3;
  TBranch                   *b_t_HitEnergies3;
  TBranch                   *b_t_DetIdEC;
  TBranch                   *b_t_HitEnergyEC;
  TBranch                   *b_t_HitDistEC;
  TBranch                   *b_t_DetIdHC;
  TBranch                   *b_t_HitEnergyHC;
  TBranch                   *b_t_HitDistHC;

  bool                       debug_;
  TFile                     *output_file;
};

CalibTreeExtended::CalibTreeExtended(const char *infile, const char *outfile,
				     const bool debug) : debug_(debug) {
  TFile      *file = new TFile(infile);
  TDirectory *dir  = (TDirectory*)(file->FindObjectAny("hcalIsoTrackStudy"));
  TTree      *tree = (TTree*)(dir->FindObjectAny("CalibTreeExtended"));
  std::cout << "Attaches tree CalibTreeExtended at " << tree << " in file " 
	    << infile << std::endl;
  
  BookHisto(outfile);
  Init(tree);
}

CalibTreeExtended::~CalibTreeExtended() {
  Close();
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t CalibTreeExtended::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t CalibTreeExtended::LoadTree(Long64_t entry) {
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

void CalibTreeExtended::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_mapP = 0;
  t_mapPt = 0;
  t_mapEta = 0;
  t_mapPhi = 0;
  t_DetIds = 0;
  t_HitEnergies = 0;
  t_trgbits = 0;
  t_DetIds1 = 0;
  t_HitEnergies1 = 0;
  t_DetIds3 = 0;
  t_HitEnergies3 = 0;
  t_DetIdEC = 0;
  t_HitEnergyEC = 0;
  t_HitDistEC = 0;
  t_DetIdHC = 0;
  t_HitEnergyHC = 0;
  t_HitDistHC = 0;
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
  fChain->SetBranchAddress("t_mapP", &t_mapP, &b_t_mapP);
  fChain->SetBranchAddress("t_mapPt", &t_mapPt, &b_t_mapPt);
  fChain->SetBranchAddress("t_mapEta", &t_mapEta, &b_t_mapEta);
  fChain->SetBranchAddress("t_mapPhi", &t_mapPhi, &b_t_mapPhi);
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_eHcal10", &t_eHcal10, &b_t_eHcal10);
  fChain->SetBranchAddress("t_eHcal30", &t_eHcal30, &b_t_eHcal30);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_emaxNearP", &t_emaxNearP, &b_t_emaxNearP);
  fChain->SetBranchAddress("t_eAnnular", &t_eAnnular, &b_t_eAnnular);
  fChain->SetBranchAddress("t_hAnnular", &t_hAnnular, &b_t_hAnnular);
  fChain->SetBranchAddress("t_rhoh", &t_rhoh, &b_t_rhoh);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyFlag", &t_qltyFlag, &b_t_qltyFlag);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_gentrackP", &t_gentrackP, &b_t_gentrackP);
  fChain->SetBranchAddress("t_gentrackE", &t_gentrackE, &b_t_gentrackE);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  fChain->SetBranchAddress("t_trgbits", &t_trgbits, &b_t_trgbits);
  fChain->SetBranchAddress("t_DetIds1", &t_DetIds1, &b_t_DetIds1);
  fChain->SetBranchAddress("t_HitEnergies1", &t_HitEnergies1, &b_t_HitEnergies1);
  fChain->SetBranchAddress("t_DetIds3", &t_DetIds3, &b_t_DetIds3);
  fChain->SetBranchAddress("t_HitEnergies3", &t_HitEnergies3, &b_t_HitEnergies3);
  fChain->SetBranchAddress("t_DetIdEC", &t_DetIdEC, &b_t_DetIdEC);
  fChain->SetBranchAddress("t_HitEnergyEC", &t_HitEnergyEC, &b_t_HitEnergyEC);
  fChain->SetBranchAddress("t_HitDistEC", &t_HitDistEC, &b_t_HitDistEC);
  fChain->SetBranchAddress("t_DetIdHC", &t_DetIdHC, &b_t_DetIdHC);
  fChain->SetBranchAddress("t_HitEnergyHC", &t_HitEnergyHC, &b_t_HitEnergyHC);
  fChain->SetBranchAddress("t_HitDistHC", &t_HitDistHC, &b_t_HitDistHC);
  Notify();
}

Bool_t CalibTreeExtended::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void CalibTreeExtended::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t CalibTreeExtended::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void CalibTreeExtended::Loop() {
  //   In a ROOT session, you can do:
  //      root> .L CalibTreeExtended.C
  //      root> CalibTreeExtended t
  //      root> t.GetEntry(12); // Fill t data members with entry number 12
  //      root> t.Show();       // Show values of entry 12
  //      root> t.Show(16);     // Read and show values of entry 16
  //      root> t.Loop();       // Loop on all entries
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
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
  }
}

void CalibTreeExtended::BookHisto(const char* fname) {
  output_file = TFile::Open(fname,"RECREATE");
}

void CalibTreeExtended::Close() {
  output_file->cd();
  if (debug_) std::cout << "file yet to be Written" << std::endl;
  output_file->Write();
  std::cout << "output file Written" << std::endl;
  output_file->Close();
  if (debug_) std::cout << "now doing return" << std::endl;
}
