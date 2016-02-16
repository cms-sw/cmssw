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
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

void Run(const char *inFileName="QCD_5_3000_PUS14",
	 const char *dirName="isopf", const char *treeName="CalibTree", 
	 const char *outFileName="QCD_5_3000_PUS14_Out", 
	 const char *corrFileName="QCD_5_3000_PUS14_Out.txt",
	 bool useweight=true, int nMin=0, bool inverse=false, 
	 double ratMin=0.25, double ratMax=10., int ietaMax=21, 
	 int applyL1Cut=1, double l1Cut=0.5);

// Fixed size dimensions of array or collections stored in the TTree if any.

class CalibTree {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           t_Run;
  Int_t           t_Event;
  Int_t           t_DataType;
  Int_t           t_ieta;
  Double_t        t_EventWeight;
  Int_t           t_goodPV;
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
  Bool_t          t_qltyFlag;
  Bool_t          t_qltyMissFlag;
  Bool_t          t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies;
  std::vector<bool>         *t_trgbits;
  std::map<unsigned int, std::pair<double,double> > Cprev;
  
  // List of branches
  TBranch        *b_t_Run;           //!
  TBranch        *b_t_Event;         //!
  TBranch        *b_t_DataType;      //!
  TBranch        *b_t_ieta;          //!
  TBranch        *b_t_EventWeight;   //!
  TBranch        *b_t_goodPV;        //!
  TBranch        *b_t_l1pt;          //!
  TBranch        *b_t_l1eta;         //!
  TBranch        *b_t_l1phi;         //!
  TBranch        *b_t_l3pt;          //!
  TBranch        *b_t_l3eta;         //!
  TBranch        *b_t_l3phi;         //!
  TBranch        *b_t_p;             //!
  TBranch        *b_t_mindR1;        //!
  TBranch        *b_t_mindR2;        //!
  TBranch        *b_t_eMipDR;        //!
  TBranch        *b_t_eHcal;         //!
  TBranch        *b_t_hmaxNearP;     //!
  TBranch        *b_t_selectTk;      //!
  TBranch        *b_t_qltyFlag;      //!
  TBranch        *b_t_qltyMissFlag;  //!
  TBranch        *b_t_qltyPVFlag;    //!
  TBranch        *b_t_DetIds;        //!
  TBranch        *b_t_HitEnergies;   //!
  TBranch        *b_t_trgbits;       //!

  TH1D     *h_pbyE;
  TProfile *h_Ebyp_bfr, *h_Ebyp_aftr;

  struct myEntry {
    myEntry (int k=0, double f0=0, double f1=0, double f2=0) : kount(k), fact0(f0),
							       fact1(f1), fact2(f2) {}
    int    kount;
    double fact0, fact1, fact2;
  };

  CalibTree(TTree *tree=0);
  virtual ~CalibTree();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual Double_t Loop(int k, TFile *fout, bool useweight, int nMin, 
			bool inverse, double rMin, double rMax, int ietaMax,
			int applyL1Cut, double l1Cut, bool last);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  bool             goodTrack();
  void             writeCorrFactor(const char *corrFileName, int ietaMax);
};


void Run(const char *inFileName, const char *dirName, const char *treeName, 
	 const char *outFileName, const char *corrFileName,  bool useweight, 
	 int nMin, bool inverse, double ratMin, double ratMax, int ietaMax, 
	 int applyL1Cut, double l1Cut) {
 
  char name[500];
  sprintf(name, "%s.root",inFileName);
  TFile *infile = TFile::Open(name);
  TDirectory *dir = (TDirectory*)infile->FindObjectAny(dirName);
  TTree *tree = (TTree*)dir->FindObjectAny(treeName);
  std::cout << "Tree " << treeName << " " << tree << " in directory " 
	    << dirName << " from file " << name << " with nentries (tracks): " 
	    << tree->GetEntries() << std::endl;
  CalibTree t(tree); 
  t.h_pbyE = new TH1D("pbyE", "pbyE", 100, -1.0, 9.0);
  t.h_Ebyp_bfr = new TProfile("Ebyp_bfr","Ebyp_bfr",60,-30,30,0,10);
  t.h_Ebyp_aftr = new TProfile("Ebyp_aftr","Ebyp_aftr",60,-30,30,0,10);
  
  sprintf(name, "%s.root",outFileName);
  TFile *fout = new TFile(outFileName, "UPDATE");
  std::cout << "Output file: " << name << " opened in update mode" << std::endl;
  fout->cd();

  double cvgs[100], itrs[100]; 
  unsigned int k(0), kmax(30);
  for (; k<=kmax; ++k) {
    std::cout << "Calling Loop() "  << k << "th time\n"; 
    double cvg = t.Loop(k, fout, useweight, nMin, inverse, ratMin, ratMax, 
			ietaMax, applyL1Cut, l1Cut, k==kmax);
    itrs[k] = k;
    cvgs[k] = cvg;
    if (cvg < 0.00001) break;
  }

  t.writeCorrFactor(corrFileName, ietaMax);

  TGraph *g_cvg;
  g_cvg = new TGraph(k, itrs, cvgs);
  fout->cd();
  g_cvg->SetMarkerStyle(7);
  g_cvg->SetMarkerSize(5.0);
  g_cvg->Draw("AP");
  g_cvg->Write("Cvg");
  std::cout << "Finish looping after " << k << " iterations" << std::endl;
  fout->Close();
}

CalibTree::CalibTree(TTree *tree) : fChain(0) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/afs/cern.ch/work/g/gwalia/public/QCD_5_3000_PUS14.root");
    if (!f || !f->IsOpen()) {
      f = new TFile("/afs/cern.ch/work/g/gwalia/public/QCD_5_3000_PUS14.root");
    }
    TDirectory * dir = (TDirectory*)f->Get("/afs/cern.ch/work/g/gwalia/public/QCD_5_3000_PUS14.root:/isopf");
    dir->GetObject("CalibTree",tree);
  }
  Init(tree);
}

CalibTree::~CalibTree() {
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
  t_trgbits = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_DataType", &t_DataType, &b_t_DataType);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_goodPV", &t_goodPV, &b_t_goodPV);
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
  fChain->SetBranchAddress("t_qltyFlag", &t_qltyFlag, &b_t_qltyFlag);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  fChain->SetBranchAddress("t_trgbits", &t_trgbits, &b_t_trgbits);
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

Double_t CalibTree::Loop(int loop, TFile *fout, bool useweight, int nMin,
			 bool inverse, double rmin, double rmax, int ietaMax,
			 int applyL1Cut, double l1Cut, bool last) {
  bool debug=false;
  if (fChain == 0) return 0;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  std::map<unsigned int, myEntry > SumW;
  std::map<unsigned int, double  > nTrks;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    //    std::cout << "***Entry (Track) Number : " << ientry << std::endl;
    //    std::cout << "p/eHCal/eMipDR/nDets : " << t_p << "/" << t_eHcal << "/"
    //	      << t_eMipDR << "/" << (*t_DetIds).size() << std::endl;
    unsigned int mask(0xFF80);
    if (goodTrack()) {
      double Etot=0.0;
      for (unsigned int idet=0; idet<(*t_DetIds).size(); idet++) { 
	double hitEn=0.0;
        unsigned int detid = (*t_DetIds)[idet] & mask;
	if (Cprev.find(detid) != Cprev.end()) 
	  hitEn = Cprev[detid].first * (*t_HitEnergies)[idet];
	else 
	  hitEn = (*t_HitEnergies)[idet];
	Etot += hitEn;
      }
      double evWt = (useweight) ? t_EventWeight : 1.0; 
      double ratio= Etot/t_p;
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
	  unsigned int detid = (*t_DetIds)[idet] & mask;
	  double hitEn=0.0;
	  if (debug) std::cout << "idet " << idet << " detid/hitenergy : " 
			       << std::hex << (*t_DetIds)[idet] << ":" 
			       << detid << "/" << (*t_HitEnergies)[idet] 
			       << std::endl;
	  if (Cprev.find(detid) != Cprev.end()) 
	    hitEn = Cprev[detid].first * (*t_HitEnergies)[idet];
	  else 
	    hitEn = (*t_HitEnergies)[idet];
	  double Wi  = evWt * hitEn/Etot;
	  double Fac = (inverse) ? (Etot/t_p) : (t_p/Etot);
	  double Fac2= Wi*Fac*Fac;
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
  if (loop==0) {
    h_pbyE->Write("h_pbyE");
    h_Ebyp_bfr->Write("h_Ebyp_bfr");
  }
  if (last) {
    h_Ebyp_aftr->Write("h_Ebyp_aftr");
  }
  std::map<unsigned int, myEntry>::iterator SumWItr = SumW.begin();
  unsigned int kount(0), kountus(0);
  double       sumfactor(0);
  double dets[150], cfacs[150], wfacs[150], myId[150], nTrk[150];
  for (; SumWItr != SumW.end(); SumWItr++) {
    unsigned int detid = SumWItr->first;
    int ieta = (detid>>7) & 0x3f;
    int zside= (detid&0x2000) ? 1 : -1;
    int depth= (detid>>14)&0x1F;
    double id = ieta*zside + 0.25*(depth-1);
    if (debug) 
      std::cout<< "Detid|kount|SumWi|SumFac|myId : " << SumWItr->first << " | "
	       << (SumWItr->second).kount << " | " << (SumWItr->second).fact0 <<"|"
	       << (SumWItr->second).fact1 << "|" << (SumWItr->second).fact2 << "|"
	       << id <<std::endl;
    double factor = (SumWItr->second).fact1/(SumWItr->second).fact0;
    double dfac   = std::sqrt(((SumWItr->second).fact2/(SumWItr->second).fact0
			       -factor*factor)/(SumWItr->second).kount);
    if (inverse) factor = 2.-factor;
    if ((SumWItr->second).kount > nMin) {
      kountus++;
      if (factor > 1) sumfactor += (1-1/factor);
      else            sumfactor += (1-factor);
    }
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
    kount++;
  }
  std::cout << kountus << " detids out of " << kount << "have trks > " << nMin << std::endl;

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
  std::map<unsigned int, std::pair<double,double> >::iterator CprevItr = Cprev.begin();
  unsigned int indx(0);
  for (; CprevItr != Cprev.end(); CprevItr++, indx++){
    unsigned int detid = CprevItr->first;
    int ieta = (detid>>7) & 0x3f;
    int zside= (detid&0x2000) ? 1 : -1;
    int depth= (detid>>14)&0x1F;
    std::cout << "DetId[" << indx << "] " << std::hex << detid << std::dec
	      << "(" << ieta*zside << "," << depth << ") ( nTrks:" 
	      << nTrks[detid] << ") : " << CprevItr->second.first << " +- "
	      << CprevItr->second.second << std::endl;
  }
  double mean = (kountus > 0) ? (sumfactor/kountus) : 0;
  std::cout << "Mean deviation " << mean << " from 1 for " << kountus 
	    << " DetIds" << std::endl;
  return mean;
}

bool CalibTree::goodTrack() {
  bool ok = ((t_selectTk) && (t_qltyMissFlag) && (t_hmaxNearP < 2.0) && 
	     (t_eMipDR < 1.0) && (t_mindR1 > 1.0) && (t_p > 40.0) &&
	     (t_p < 60.0));
  return ok;
}

void CalibTree::writeCorrFactor(const char *corrFileName, int ietaMax) {
  ofstream myfile;
  myfile.open(corrFileName);
  if (!myfile.is_open()) {
    std::cout << "** ERROR: Can't open '" << corrFileName << std::endl;
  } else {
    myfile << std::setprecision(4) << std::setw(10) << "detId" 
	   << std::setw(10) << "ieta" << std::setw(10) << "depth" 
	   << std::setw(10) << "corrFactor" << std::endl;
    std::map<unsigned int, std::pair<double,double> >::const_iterator itr;
    for (itr=Cprev.begin(); itr != Cprev.end(); ++itr) {
      unsigned int detId = itr->first;
      int etaAbs= ((detId>>7)&0x3f);
      int ieta  = ((detId&0x2000) ? etaAbs : -etaAbs);
      int depth = ((detId>>14)&0x1f);
      if (etaAbs <= ietaMax) {
	myfile << std::setw(10) << std::hex << detId << std::setw(10) 
	       << std::dec << ieta << std::setw(10) << depth << std::setw(10) 
	       << itr->second.first << " " << std::setw(10) << itr->second.second 
	       << std::endl;
      }
    }
    myfile.close();
  }
}
