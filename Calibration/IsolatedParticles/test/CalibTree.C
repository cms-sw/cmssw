//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Sep  9 17:21:19 2014 by ROOT version 5.34/03
// from TTree CalibTree/CalibTree
// found on file: output.root
//////////////////////////////////////////////////////////

#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TGraph.h>
#include <TProfile.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

void Run(//const char *inFileName="root://eoscms//eos/cms/store/caf/user/gwalia/QCD_5_1000_S14",
	 const char *inFileName="QCD_5_1000_S14.root",
	 const char *outFileName="CalibHisto",
	 const char *dirname="isopf", const char *treeName="CalibTree");

// Fixed size dimensions of array or collections stored in the TTree if any.

class CalibTree {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           t_Run;
  Int_t           t_Event;
  Int_t           t_ieta;
  Double_t        t_EventWeight;
  Double_t        t_l1pt;
  Double_t        t_l1eta;
  Double_t        t_l1phi;
  Double_t        t_l3pt;
  Double_t        t_l3eta;
  Double_t        t_l3phi;
  Double_t        t_p;
  Double_t        t_mindR1;
  Double_t        t_mindR2;
  Double_t        t_eMipDR;
  Double_t        t_eHcal;
  Double_t        t_hmaxNearP;
  Bool_t          t_selectTk;
  Bool_t          t_qltyMissFlag;
  Bool_t          t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>  *t_HitEnergies;
  std::map<unsigned int, double> Cprev;
  
  // List of branches
  TBranch        *b_t_Run;   //!
  TBranch        *b_t_Event;   //!
  TBranch        *b_t_ieta;   //!
  TBranch        *b_t_EventWeight;   //!
  TBranch        *b_t_l1pt;   //!
  TBranch        *b_t_l1eta;   //!
  TBranch        *b_t_l1phi;   //!
  TBranch        *b_t_l3pt;   //!
  TBranch        *b_t_l3eta;   //!
  TBranch        *b_t_l3phi;   //!
  TBranch        *b_t_p;   //!
  TBranch        *b_t_mindR1;   //!
  TBranch        *b_t_mindR2;   //!
  TBranch        *b_t_eMipDR;   //!
  TBranch        *b_t_eHcal;   //!
  TBranch        *b_t_hmaxNearP;   //!
  TBranch        *b_t_selectTk;   //!
  TBranch        *b_t_qltyMissFlag;   //!
  TBranch        *b_t_qltyPVFlag;   //!
  TBranch        *b_t_DetIds;   //!
  TBranch        *b_t_HitEnergies;   //!

  CalibTree(TTree *tree=0);
  virtual ~CalibTree();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual Double_t Loop(int loop);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  bool             goodTrack();
  void BookHisto(std::string fname);
  TFile          *fout;

private:
  TProfile       *hprof_ndets;

};

void Run( const char *inFileName, const char *outFileName, 
	  const char *dirname, const char *treeName) {  
  char name[500];
  sprintf(name, "%s.root",inFileName);
  TFile *infile = TFile::Open(name);
  TDirectory *dir = (TDirectory*)infile->FindObjectAny(dirname);
  TTree *tree = (TTree*)dir->FindObjectAny(treeName);
  std::cout << tree << " tree with nentries (tracks): " << tree->GetEntries() << std::endl;
  CalibTree t(tree);
  sprintf(name, "%s_%s_%s.root", outFileName, inFileName, dirname);
  std::string outFile(name);
  t.BookHisto(outFile);

  double cvgs[100], itrs[100]; 
  unsigned int k(0);
  for (; k<20; ++k) {
    double cvg = t.Loop(k);
    itrs[k] = k;
    cvgs[k] = cvg;
    //    if (cvg < 0.00001) break;
  }
  TGraph *g_cvg;
  g_cvg = new TGraph(k, itrs, cvgs);
  t.fout->WriteTObject(g_cvg, "g_cvg");
  std::cout << "Finish looping after " << k << " iterations" << std::endl;
}

CalibTree::CalibTree(TTree *tree) : fChain(0) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("output.root");
    if (!f || !f->IsOpen()) {
      f = new TFile("output.root");
    }
    TDirectory * dir = (TDirectory*)f->Get("output.root:/isopf");
    dir->GetObject("CalibTree",tree);
  }
  Init(tree);
}

CalibTree::~CalibTree() {
  fout->cd();
  fout->Write();
  fout->Close();
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

void CalibTree::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_DetIds = 0;
  t_HitEnergies = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  Notify();
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

Double_t CalibTree::Loop(int loop) {
  char name[500];
  bool debug=false;
  if (fChain == 0) return 0;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  std::map<unsigned int, std::pair<double,double> >SumW;
  std::map<unsigned int, unsigned int >nTrks;
  unsigned int mask(0xFF80), ntrkMax(0);
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
  //  for (Long64_t jentry=0; jentry<1000;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if(debug) std::cout << "***Entry (Track) Number : " << ientry 
			<< " p/eHCal/eMipDR/nDets : " << t_p << "/" << t_eHcal 
			<< "/" << t_eMipDR << "/" << (*t_DetIds).size() 
			<< std::endl;
    if (goodTrack()) {
      if (loop == 0) hprof_ndets->Fill(t_ieta, (*t_DetIds).size());
      double Etot=0.0;
      for (unsigned int idet=0; idet<(*t_DetIds).size(); idet++) { 
	double hitEn=0.0;
        unsigned int detid = (*t_DetIds)[idet] & mask;
	if (Cprev.find(detid) != Cprev.end()) 
	  hitEn = Cprev[detid] * (*t_HitEnergies)[idet];
	else 
	  hitEn = (*t_HitEnergies)[idet];
	Etot += hitEn;
      }
      for (unsigned int idet=0; idet<(*t_DetIds).size(); idet++) {
	unsigned int detid = (*t_DetIds)[idet] & mask;
	double hitEn=0.0;
	if (debug) std::cout << "idet " << idet << " detid/hitenergy : " 
			     << std::hex << (*t_DetIds)[idet] << ":" << detid
			     << "/" << (*t_HitEnergies)[idet] << std::endl;
	if (Cprev.find(detid) != Cprev.end()) 
	  hitEn = Cprev[detid] * (*t_HitEnergies)[idet];
	else 
	  hitEn = (*t_HitEnergies)[idet];
	double Wi = hitEn/Etot;
	double Fac = (Wi* t_p) / Etot;
	if( SumW.find(detid) != SumW.end() ) {
	  Wi  += SumW[detid].first;
	  Fac += SumW[detid].second;
	  SumW[detid] = std::pair<double,double>(Wi,Fac); 
	  nTrks[detid]++;
	} else {
	  SumW.insert( std::pair<unsigned int, std::pair<double,double> >(detid,std::pair<double,double>(Wi,Fac)));
	  nTrks.insert(std::pair<unsigned int,unsigned int>(detid, 1));
	}
	if (nTrks[detid] > ntrkMax) ntrkMax = nTrks[detid];
      }
    }
  }
  
  std::map<unsigned int, std::pair<double,double> >::iterator SumWItr = SumW.begin();
  unsigned int kount(0), mkount(0);
  double       sumfactor(0);
  double       dets[150], cfacs[150], wfacs[150], nTrk[150];
  unsigned int ntrkCut = ntrkMax/10;
  for (; SumWItr != SumW.end(); SumWItr++) {
    if (debug) 
      std::cout<< "Detid/SumWi/SumFac : " << SumWItr->first << " / "
	       << (SumWItr->second).first << " / " << (SumWItr->second).second
	       << std::endl;
    unsigned int detid = SumWItr->first;
    double factor = (SumWItr->second).second / (SumWItr->second).first;

    if(nTrks[detid]>ntrkCut) {
      if (factor > 1) sumfactor += (1-1/factor);
      else            sumfactor += (1-factor);
      mkount++;
    }
    if (Cprev.find(detid) != Cprev.end()) {
      Cprev[detid] *= factor;
      cfacs[kount] = Cprev[detid];
    } else {
      Cprev.insert( std::pair<unsigned int, double>(detid, factor) );
      cfacs[kount] = factor;
    }
    int ieta = (detid>>7) & 0x3f;
    int zside= (detid&0x2000) ? 1 : -1;
    int depth= (detid>>14)&0x1F;
    wfacs[kount]= factor;
    dets[kount] = zside*(ieta+0.1*(depth-1));
    nTrk[kount] = nTrks[detid];
    kount++;
  }
  TGraph *g_fac, *g_fac2, *g_nTrk;
  g_fac = new TGraph(kount, dets, cfacs); 
  sprintf(name, "Cfacs_detid_it%d", loop);
  fout->WriteTObject(g_fac, name);

  g_fac2 = new TGraph(kount, dets, wfacs); 
  sprintf(name, "Wfacs_detid_it%d", loop);
  fout->WriteTObject(g_fac2, name);

  g_nTrk = new TGraph(kount, dets, nTrk); 
  if (loop==0) fout->WriteTObject(g_nTrk, "nTrk_detid");

  std::cout << "The new factors are :" << std::endl;
  std::map<unsigned int, double>::iterator CprevItr = Cprev.begin();
  unsigned int indx(0);
  for (CprevItr=Cprev.begin(); CprevItr != Cprev.end(); CprevItr++, indx++){
    unsigned int detid = CprevItr->first;
    int ieta = (detid>>7) & 0x3f;
    int zside= (detid&0x2000) ? 1 : -1;
    int depth= (detid>>14)&0x1F;
    std::cout << "DetId[" << indx << "] " << std::hex << detid << std::dec
	      << "(" << ieta*zside << "," << depth << ") ( nTrks:" 
	      << nTrks[detid] << ") : " << CprevItr->second << std::endl;
  }
  double mean = (mkount > 0) ? (sumfactor/mkount) : 0;
  std::cout << "Mean deviation " << mean << " from 1 for " << mkount << ":"
	    << kount << " DetIds" << std::endl;
  return mean;
}

bool CalibTree::goodTrack() {
  bool ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < 2.0) && 
	     (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (t_p > 40.0) &&
	     (t_p < 60.0) && (t_eHcal > 0.5*t_p));
  return ok;
}

void CalibTree::BookHisto(std::string fname){
  fout = new TFile(fname.c_str(), "UPDATE");
  fout->cd();
  hprof_ndets  = new TProfile("hndets","number of detids versus ieta",60,-30,30,0,20);

}
