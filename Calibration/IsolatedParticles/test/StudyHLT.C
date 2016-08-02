//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jun  2 08:48:56 2016 by ROOT version 5.34/34
// from TTree testTree/new HLT Tree
// found on file: temp.root
//////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <vector>
#include <iostream>

class StudyHLT {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t                       tr_goodRun;
  Int_t                       tr_goodPV;
  std::vector<std::string>   *tr_TrigName;
  std::vector<double>        *tr_TrkPt;
  std::vector<double>        *tr_TrkP;
  std::vector<double>        *tr_TrkEta;
  std::vector<double>        *tr_TrkPhi;
  std::vector<double>        *tr_MaxNearP31X31;
  std::vector<double>        *tr_MaxNearHcalP7x7;
  std::vector<double>        *tr_FE7x7P;
  std::vector<double>        *tr_FE11x11P;
  std::vector<double>        *tr_FE15x15P;
  std::vector<bool>          *tr_SE7x7P;
  std::vector<bool>          *tr_SE11x11P;
  std::vector<bool>          *tr_SE15x15P;
  std::vector<double>        *tr_H7x7;
  std::vector<double>        *tr_H5x5;
  std::vector<double>        *tr_H3x3;
  std::vector<int>           *tr_iEta;

  // List of branches
  TBranch                    *b_tr_goodRun;   //!
  TBranch                    *b_tr_goodPV;   //!
  TBranch                    *b_tr_TrigName;   //!
  TBranch                    *b_tr_TrkPt;   //!
  TBranch                    *b_tr_TrkP;   //!
  TBranch                    *b_tr_TrkEta;   //!
  TBranch                    *b_tr_TrkPhi;   //!
  TBranch                    *b_tr_MaxNearP31X31;   //!
  TBranch                    *b_tr_MaxNearHcalP7x7;   //!
  TBranch                    *b_tr_TrkQuality;   //!
  TBranch                    *b_tr_TrkokECAL;   //!
  TBranch                    *b_tr_TrkokHCAL;   //!
  TBranch                    *b_tr_FE7x7P;   //!
  TBranch                    *b_tr_FE11x11P;   //!
  TBranch                    *b_tr_FE15x15P;   //!
  TBranch                    *b_tr_SE7x7P;   //!
  TBranch                    *b_tr_SE11x11P;   //!
  TBranch                    *b_tr_SE15x15P;   //!
  TBranch                    *b_tr_H7x7;   //!
  TBranch                    *b_tr_H5x5;   //!
  TBranch                    *b_tr_H3x3;   //!
  TBranch                    *b_tr_iEta;   //!

  StudyHLT(std::string inFile, std::string dirnam, std::string treeNam,
	   std::string outFile);
  virtual ~StudyHLT();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

  std::string      outFile_;
};

StudyHLT::StudyHLT(std::string inFile, std::string dirnam, std::string treeNam,
		   std::string outFile) : outFile_(outFile) {

  TFile      *file = new TFile(inFile.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnam.c_str());
  std::cout << inFile << " file " << file << " " << dirnam << " " << dir 
	    << std::endl;
  TTree      *tree = (TTree*)dir->Get(treeNam.c_str());
  std::cout << "Tree:" << treeNam << " at " << tree << std::endl;
  Init(tree);
}

StudyHLT::~StudyHLT() {
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t StudyHLT::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t StudyHLT::LoadTree(Long64_t entry) {
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

void StudyHLT::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  tr_TrigName = 0;
  tr_TrkPt = 0;
  tr_TrkP = 0;
  tr_TrkEta = 0;
  tr_TrkPhi = 0;
  tr_MaxNearP31X31 = 0;
  tr_MaxNearHcalP7x7 = 0;
  tr_FE7x7P = 0;
  tr_FE11x11P = 0;
  tr_FE15x15P = 0;
  tr_SE7x7P = 0;
  tr_SE11x11P = 0;
  tr_SE15x15P = 0;
  tr_H3x3 = 0;
  tr_H5x5 = 0;
  tr_H7x7 = 0;

  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("tr_goodRun",  &tr_goodRun,  &b_tr_goodRun);
  fChain->SetBranchAddress("tr_goodPV",   &tr_goodPV,   &b_tr_goodPV);
  fChain->SetBranchAddress("tr_TrigName", &tr_TrigName, &b_tr_TrigName);
  fChain->SetBranchAddress("tr_TrkPt",    &tr_TrkPt,    &b_tr_TrkPt);
  fChain->SetBranchAddress("tr_TrkP",     &tr_TrkP,     &b_tr_TrkP);
  fChain->SetBranchAddress("tr_TrkEta",   &tr_TrkEta,   &b_tr_TrkEta);
  fChain->SetBranchAddress("tr_TrkPhi",   &tr_TrkPhi,   &b_tr_TrkPhi);
  fChain->SetBranchAddress("tr_MaxNearP31X31",   &tr_MaxNearP31X31,   &b_tr_MaxNearP31X31);
  fChain->SetBranchAddress("tr_MaxNearHcalP7x7", &tr_MaxNearHcalP7x7, &b_tr_MaxNearHcalP7x7);
  fChain->SetBranchAddress("tr_FE7x7P",   &tr_FE7x7P,   &b_tr_FE7x7P);
  fChain->SetBranchAddress("tr_FE11x11P", &tr_FE11x11P, &b_tr_FE11x11P);
  fChain->SetBranchAddress("tr_FE15x15P", &tr_FE15x15P, &b_tr_FE15x15P);
  fChain->SetBranchAddress("tr_SE7x7P",   &tr_SE7x7P,   &b_tr_SE7x7P);
  fChain->SetBranchAddress("tr_SE11x11P", &tr_SE11x11P, &b_tr_SE11x11P);
  fChain->SetBranchAddress("tr_SE15x15P", &tr_SE15x15P, &b_tr_SE15x15P);
  fChain->SetBranchAddress("tr_H3x3",     &tr_H3x3,     &b_tr_H3x3);
  fChain->SetBranchAddress("tr_H5x5",     &tr_H5x5,     &b_tr_H5x5);
  fChain->SetBranchAddress("tr_H7x7",     &tr_H7x7,     &b_tr_H7x7);
  fChain->SetBranchAddress("tr_iEta",     &tr_iEta,     &b_tr_iEta);
  Notify();
}

Bool_t StudyHLT::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void StudyHLT::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t StudyHLT::Cut(Long64_t ) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void StudyHLT::Loop() {

  //create a new root file to store the output.
  TFile *f = new TFile(outFile_.c_str(), "RECREATE");
  if (fChain == 0) return;

  //declare the histograms in the same way as is done in studyHLT
  std::string titls[6] = {"NoIso","okEcal","EcalCharIso","HcalCharIso",
			  "EcalNeutIso","HcalNeutIso"};
  char name[40], htit[400];
  TH1D   *h_pt[6], *h_p[6], *h_eta[6], *h_phi[6];
  for (int i=0; i<6; ++i) {
    sprintf (name, "h_pt_%s", titls[i].c_str());
    h_pt[i]  = new TH1D(name,"", 400, 0, 200);
    sprintf (name, "h_p_%s", titls[i].c_str());
    h_p[i]   = new TH1D(name,"", 400, 0, 200);
    sprintf (name, "h_eta_%s", titls[i].c_str());
    h_eta[i] = new TH1D(name,"", 60, -3.0, 3.0);
    sprintf (name, "h_phi_%s", titls[i].c_str());
    h_phi[i] = new TH1D(name,"", 100, -3.15, 3.15);
  }
  static const int           nPBin=10, nEtaBin=4, nPVBin=4;
  TH1D                      *h_energy[nPVBin+1][nPBin][nEtaBin][6];
  double pBin[nPBin+1] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,9.0,11.0,15.0,20.0};
  int    etaBin[nEtaBin+1] = {1, 7, 13, 17, 23};
  int    pvBin[nPVBin+1] = {1, 2, 3, 5, 100};
  std::string energyNames[6]={"E_{7x7}", "H_{3x3}", "(E_{7x7}+H_{3x3})",
			      "E_{11x11}", "H_{5x5}", "{E_{11x11}+H_{5x5})"};
  for (int i=0; i<nPVBin; ++i) {
    for (int ip=0; ip<nPBin; ++ip) {
      for (int ie=0; ie<nEtaBin; ++ie) {
	for (int j=0; j<6; ++j) {
	  sprintf(name, "h_energy_%d_%d_%d_%d", i, ip, ie, j);
	  if (i < nPVBin) {
	    sprintf(htit,"%s/p (p=%4.1f:%4.1f; i#eta=%d:%d, PV=%d:%d)",
		    energyNames[j].c_str(),pBin[ip],pBin[ip+1],etaBin[ie],
		    (etaBin[ie+1]-1), pvBin[i], pvBin[i+1]);
	  } else {
	    sprintf(htit,"%s/p (p=%4.1f:%4.1f; i#eta=%d:%d, All PV)",
		    energyNames[j].c_str(),pBin[ip],pBin[ip+1],etaBin[ie],
		    (etaBin[ie+1]-1));
	  }
	  h_energy[i][ip][ie][j] = new TH1D(name, htit, 500, -0.1, 4.9);
	  h_energy[i][ip][ie][j]->Sumw2();
	}
      }
    }
  }
  Long64_t nentries = fChain->GetEntries();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    if ((tr_TrkPt->size() != tr_TrkP->size()) || 
	(tr_TrkPt->size() != tr_TrkEta->size()) || 
	(tr_TrkPt->size() != tr_TrkPhi->size())) {
      std::cout << "Error processing samples " << std::endl;
      break;
    }//matches if

    //fill the distributions
    //loop over all the reco tracks
    for (unsigned int itk=0; itk !=tr_TrkPt->size(); ++itk){

      h_pt[0]->Fill(  ((*tr_TrkPt)[itk]));
      h_p[0]->Fill(   ((*tr_TrkP)[itk]));
      h_eta[0]->Fill( ((*tr_TrkEta)[itk]));
      h_phi[0]->Fill( ((*tr_TrkPhi)[itk]));
      if ( (*tr_TrkPt)[itk] > 1.0 && abs((*tr_TrkEta)[itk]) < 2.5) {
	h_pt[1]->Fill(  ((*tr_TrkPt)[itk]));
	h_p[1]->Fill(   ((*tr_TrkP)[itk]));
	h_eta[1]->Fill( ((*tr_TrkEta)[itk]));
	h_phi[1]->Fill( ((*tr_TrkPhi)[itk]));
	//condition of charged Isolation
	if ( (*tr_MaxNearP31X31)[itk] < 0) {
	  h_pt[2]->Fill(  ((*tr_TrkPt)[itk]));
	  h_p[2]->Fill(   ((*tr_TrkP)[itk]));
	  h_eta[2]->Fill( ((*tr_TrkEta)[itk]));
	  h_phi[2]->Fill( ((*tr_TrkPhi)[itk]));
	  //condition for HCal Charged Isolation
	  if ( (*tr_MaxNearHcalP7x7)[itk] < 0) {
	    h_pt[3]->Fill(  ((*tr_TrkPt)[itk]));
	    h_p[3]->Fill(   ((*tr_TrkP)[itk]));
	    h_eta[3]->Fill( ((*tr_TrkEta)[itk]));
	    h_phi[3]->Fill( ((*tr_TrkPhi)[itk]));
	    //condition of Neutral Isolation
	    if ( (*tr_SE11x11P)[itk] && (*tr_SE15x15P)[itk] && 
		 ( (*tr_FE15x15P)[itk] - (*tr_FE11x11P)[itk]) < 2.0 ) {
	      h_pt[4]->Fill(  ((*tr_TrkPt)[itk]));
	      h_p[4]->Fill(   ((*tr_TrkP)[itk]));
	      h_eta[4]->Fill( ((*tr_TrkEta)[itk]));
	      h_phi[4]->Fill( ((*tr_TrkPhi)[itk]));
	      if ( ((*tr_H7x7)[itk] - (*tr_H5x5)[itk] ) < 2.0) {
		h_pt[5]->Fill(  ((*tr_TrkPt)[itk]));
		h_p[5]->Fill(   ((*tr_TrkP)[itk]));
		h_eta[5]->Fill( ((*tr_TrkEta)[itk]));
		h_phi[5]->Fill( ((*tr_TrkPhi)[itk]));
		int ip(-1), ie(-1), nPV(-1);
		for (int i=0; i<nPBin; ++i) {
		  if (((*tr_TrkP)[itk] >= pBin[i]) && 
		      ((*tr_TrkP)[itk] < pBin[i+1])) { ip = i; break; }
		}
		for (int i=0; i<nEtaBin; ++i) {
		  if (((*tr_iEta)[itk] >= etaBin[i]) && 
		      ((*tr_iEta)[itk] < etaBin[i+1])) { ie = i; break; }
		}
		for (int i=0; i<nPVBin; ++i) {
		  if (((tr_goodPV) >= pvBin[i]) && 
		      ((tr_goodPV) < pvBin[i+1])) { nPV = i; break; }
		}
		double den = 1.0/((*tr_TrkP)[itk]);
		if ((ip >= 0) && (ie >= 0) && ((*tr_FE7x7P)[itk] > 0.02)
		    && ((*tr_H3x3)[itk] > 0.1) && (nPV >= 0)) {
		  h_energy[nPV][ip][ie][0]->Fill(den*(*tr_FE7x7P)[itk]);
		  h_energy[nPV][ip][ie][1]->Fill(den*(*tr_H3x3)[itk]);
		  h_energy[nPV][ip][ie][2]->Fill(den*((*tr_FE7x7P)[itk]+(*tr_H3x3)[itk]));
		  h_energy[nPV][ip][ie][3]->Fill(den*(*tr_FE11x11P)[itk]);
		  h_energy[nPV][ip][ie][4]->Fill(den*(*tr_H5x5)[itk]);
		  h_energy[nPV][ip][ie][5]->Fill(den*((*tr_FE11x11P)[itk]+(*tr_H5x5)[itk]));
		  h_energy[nPVBin][ip][ie][0]->Fill(den*(*tr_FE7x7P)[itk]);
		  h_energy[nPVBin][ip][ie][1]->Fill(den*(*tr_H3x3)[itk]);
		  h_energy[nPVBin][ip][ie][2]->Fill(den*((*tr_FE7x7P)[itk]+(*tr_H3x3)[itk]));
		  h_energy[nPVBin][ip][ie][3]->Fill(den*(*tr_FE11x11P)[itk]);
		  h_energy[nPVBin][ip][ie][4]->Fill(den*(*tr_H5x5)[itk]);
		  h_energy[nPVBin][ip][ie][5]->Fill(den*((*tr_FE11x11P)[itk]+(*tr_H5x5)[itk]));
		}
	      }//HCal Neutral Iso
	    }//neutral isolation 
	  }//HCal Charged Iso
	}//charged Iso
      }
    }//end for loop
  }
  f->Write();
  f->Close();
}

