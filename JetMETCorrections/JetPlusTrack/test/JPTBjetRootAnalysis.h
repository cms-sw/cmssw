//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon May 18 14:41:05 2009 by ROOT version 5.18/00a
// from TTree t1/analysis tree
// found on file: jptbjetanalyzer.root
//////////////////////////////////////////////////////////

#ifndef JPTBjetRootAnalysis_h
#define JPTBjetRootAnalysis_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

class JPTBjetRootAnalysis {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Double_t        EtaGen1;
   Double_t        PhiGen1;
   Int_t           ElectronFlagGen1;
   Int_t           MuonFlagGen1;
   Int_t           TauFlagGen1;
   Int_t           ElectronFlagGen1NoLep;
   Int_t           MuonFlagGen1NoLep;
   Int_t           TauFlagGen1NoLep;
   Int_t           BJet1FlagGen;
   Double_t        EtaRaw1;
   Double_t        PhiRaw1;
   Double_t        EtGen1;
   Double_t        EtRaw1;
   Double_t        EtMCJ1;
   Double_t        EtZSP1;
   Double_t        EtJPT1;
   Double_t        hoE;
   Double_t        DRMAXgjet1;
   Double_t        drElecFromZjet1;
   Double_t        drMuonFromZjet1;
   Double_t        drTauFromZjet1;
   Double_t        EtaGen2;
   Double_t        PhiGen2;
   Int_t           ElectronFlagGen2;
   Int_t           MuonFlagGen2;
   Int_t           TauFlagGen2;
   Int_t           ElectronFlagGen2NoLep;
   Int_t           MuonFlagGen2NoLep;
   Int_t           TauFlagGen2NoLep;
   Int_t           BJet2FlagGen;
   Double_t        EtaRaw2;
   Double_t        PhiRaw2;
   Double_t        EtGen2;
   Double_t        EtRaw2;
   Double_t        EtMCJ2;
   Double_t        EtZSP2;
   Double_t        EtJPT2;
   Double_t        DRMAXgjet2;
   Double_t        drElecFromZjet2;
   Double_t        drMuonFromZjet2;
   Double_t        drTauFromZjet2;
   Int_t           nelecs;
   Double_t        elecMom[10];   //[nelecs]
   Double_t        elecPt[10];   //[nelecs]
   Int_t           nmuons;
   Double_t        muonMom[10];   //[nmuons]
   Double_t        muonPt[10];   //[nmuons]
   Int_t           ntaus;
   Double_t        tauMom[10];   //[ntaus]
   Double_t        tauPt[10];   //[ntaus]

   // List of branches
   TBranch        *b_EtaGen1;   //!
   TBranch        *b_PhiGen1;   //!
   TBranch        *b_ElectronFlagGen1;   //!
   TBranch        *b_MuonFlagGen1;   //!
   TBranch        *b_TauFlagGen1;   //!
   TBranch        *b_ElectronFlagGen1NoLep;   //!
   TBranch        *b_MuonFlagGen1NoLep;   //!
   TBranch        *b_TauFlagGen1NoLep;   //!
   TBranch        *b_BJet1FlagGen;   //!
   TBranch        *b_EtaRaw1;   //!
   TBranch        *b_PhiRaw1;   //!
   TBranch        *b_EtGen1;   //!
   TBranch        *b_EtRaw1;   //!
   TBranch        *b_EtMCJ1;   //!
   TBranch        *b_EtZSP1;   //!
   TBranch        *b_EtJPT1;   //!
   TBranch        *b_hoE;   //!
   TBranch        *b_DRMAXgjet1;   //!
   TBranch        *b_drElecFromZjet1;   //!
   TBranch        *b_drMuonFromZjet1;   //!
   TBranch        *b_drTauFromZjet1;   //!
   TBranch        *b_EtaGen2;   //!
   TBranch        *b_PhiGen2;   //!
   TBranch        *b_ElectronFlagGen2;   //!
   TBranch        *b_MuonFlagGen2;   //!
   TBranch        *b_TauFlagGen2;   //!
   TBranch        *b_ElectronFlagGen2NoLep;   //!
   TBranch        *b_MuonFlagGen2NoLep;   //!
   TBranch        *b_TauFlagGen2NoLep;   //!
   TBranch        *b_BJet2FlagGen;   //!
   TBranch        *b_EtaRaw2;   //!
   TBranch        *b_PhiRaw2;   //!
   TBranch        *b_EtGen2;   //!
   TBranch        *b_EtRaw2;   //!
   TBranch        *b_EtMCJ2;   //!
   TBranch        *b_EtZSP2;   //!
   TBranch        *b_EtJPT2;   //!
   TBranch        *b_DRMAXgjet2;   //!
   TBranch        *b_drElecFromZjet2;   //!
   TBranch        *b_drMuonFromZjet2;   //!
   TBranch        *b_drTauFromZjet2;   //!
   TBranch        *b_nelecs;   //!
   TBranch        *b_elecMom;   //!
   TBranch        *b_elecPt;   //!
   TBranch        *b_nmuons;   //!
   TBranch        *b_muonMom;   //!
   TBranch        *b_muonPt;   //!
   TBranch        *b_ntaus;   //!
   TBranch        *b_tauMom;   //!
   TBranch        *b_tauPt;   //!

   JPTBjetRootAnalysis(TTree *tree=0);
   virtual ~JPTBjetRootAnalysis();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual void     Reset();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef JPTBjetRootAnalysis_cxx
JPTBjetRootAnalysis::JPTBjetRootAnalysis(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("jptbjetanalyzer.root");
      if (!f) {
         f = new TFile("jptbjetanalyzer.root");
      }
      tree = (TTree*)gDirectory->Get("t1");

   }
   Init(tree);
}

JPTBjetRootAnalysis::~JPTBjetRootAnalysis()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t JPTBjetRootAnalysis::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t JPTBjetRootAnalysis::LoadTree(Long64_t entry)
{
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

void JPTBjetRootAnalysis::Init(TTree *tree)
{
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

   fChain->SetBranchAddress("EtaGen1", &EtaGen1, &b_EtaGen1);
   fChain->SetBranchAddress("PhiGen1", &PhiGen1, &b_PhiGen1);
   fChain->SetBranchAddress("ElectronFlagGen1", &ElectronFlagGen1, &b_ElectronFlagGen1);
   fChain->SetBranchAddress("MuonFlagGen1", &MuonFlagGen1, &b_MuonFlagGen1);
   fChain->SetBranchAddress("TauFlagGen1", &TauFlagGen1, &b_TauFlagGen1);
   fChain->SetBranchAddress("ElectronFlagGen1NoLep", &ElectronFlagGen1NoLep, &b_ElectronFlagGen1NoLep);
   fChain->SetBranchAddress("MuonFlagGen1NoLep", &MuonFlagGen1NoLep, &b_MuonFlagGen1NoLep);
   fChain->SetBranchAddress("TauFlagGen1NoLep", &TauFlagGen1NoLep, &b_TauFlagGen1NoLep);
   fChain->SetBranchAddress("BJet1FlagGen", &BJet1FlagGen, &b_BJet1FlagGen);
   fChain->SetBranchAddress("EtaRaw1", &EtaRaw1, &b_EtaRaw1);
   fChain->SetBranchAddress("PhiRaw1", &PhiRaw1, &b_PhiRaw1);
   fChain->SetBranchAddress("EtGen1", &EtGen1, &b_EtGen1);
   fChain->SetBranchAddress("EtRaw1", &EtRaw1, &b_EtRaw1);
   fChain->SetBranchAddress("EtMCJ1", &EtMCJ1, &b_EtMCJ1);
   fChain->SetBranchAddress("EtZSP1", &EtZSP1, &b_EtZSP1);
   fChain->SetBranchAddress("EtJPT1", &EtJPT1, &b_EtJPT1);
   fChain->SetBranchAddress("hoE", &hoE, &b_hoE);
   fChain->SetBranchAddress("DRMAXgjet1", &DRMAXgjet1, &b_DRMAXgjet1);
   fChain->SetBranchAddress("drElecFromZjet1", &drElecFromZjet1, &b_drElecFromZjet1);
   fChain->SetBranchAddress("drMuonFromZjet1", &drMuonFromZjet1, &b_drMuonFromZjet1);
   fChain->SetBranchAddress("drTauFromZjet1", &drTauFromZjet1, &b_drTauFromZjet1);
   fChain->SetBranchAddress("EtaGen2", &EtaGen2, &b_EtaGen2);
   fChain->SetBranchAddress("PhiGen2", &PhiGen2, &b_PhiGen2);
   fChain->SetBranchAddress("ElectronFlagGen2", &ElectronFlagGen2, &b_ElectronFlagGen2);
   fChain->SetBranchAddress("MuonFlagGen2", &MuonFlagGen2, &b_MuonFlagGen2);
   fChain->SetBranchAddress("TauFlagGen2", &TauFlagGen2, &b_TauFlagGen2);
   fChain->SetBranchAddress("ElectronFlagGen2NoLep", &ElectronFlagGen2NoLep, &b_ElectronFlagGen2NoLep);
   fChain->SetBranchAddress("MuonFlagGen2NoLep", &MuonFlagGen2NoLep, &b_MuonFlagGen2NoLep);
   fChain->SetBranchAddress("TauFlagGen2NoLep", &TauFlagGen2NoLep, &b_TauFlagGen2NoLep);
   fChain->SetBranchAddress("BJet2FlagGen", &BJet2FlagGen, &b_BJet2FlagGen);
   fChain->SetBranchAddress("EtaRaw2", &EtaRaw2, &b_EtaRaw2);
   fChain->SetBranchAddress("PhiRaw2", &PhiRaw2, &b_PhiRaw2);
   fChain->SetBranchAddress("EtGen2", &EtGen2, &b_EtGen2);
   fChain->SetBranchAddress("EtRaw2", &EtRaw2, &b_EtRaw2);
   fChain->SetBranchAddress("EtMCJ2", &EtMCJ2, &b_EtMCJ2);
   fChain->SetBranchAddress("EtZSP2", &EtZSP2, &b_EtZSP2);
   fChain->SetBranchAddress("EtJPT2", &EtJPT2, &b_EtJPT2);
   fChain->SetBranchAddress("DRMAXgjet2", &DRMAXgjet2, &b_DRMAXgjet2);
   fChain->SetBranchAddress("drElecFromZjet2", &drElecFromZjet2, &b_drElecFromZjet2);
   fChain->SetBranchAddress("drMuonFromZjet2", &drMuonFromZjet2, &b_drMuonFromZjet2);
   fChain->SetBranchAddress("drTauFromZjet2", &drTauFromZjet2, &b_drTauFromZjet2);
   fChain->SetBranchAddress("nelecs", &nelecs, &b_nelecs);
   fChain->SetBranchAddress("elecMom", elecMom, &b_elecMom);
   fChain->SetBranchAddress("elecPt", elecPt, &b_elecPt);
   fChain->SetBranchAddress("nmuons", &nmuons, &b_nmuons);
   fChain->SetBranchAddress("muonMom", muonMom, &b_muonMom);
   fChain->SetBranchAddress("muonPt", muonPt, &b_muonPt);
   fChain->SetBranchAddress("ntaus", &ntaus, &b_ntaus);
   fChain->SetBranchAddress("tauMom", tauMom, &b_tauMom);
   fChain->SetBranchAddress("tauPt", tauPt, &b_tauPt);
   Notify();
}

Bool_t JPTBjetRootAnalysis::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void JPTBjetRootAnalysis::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t JPTBjetRootAnalysis::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
void JPTBjetRootAnalysis::Reset()
{

  // initialize tree variables
  EtaGen1 = 0.;
  PhiGen1 = 0.;
  EtaRaw1 = 0.;
  PhiRaw1 = 0.;
  EtGen1  = 0.;
  EtRaw1  = 0.;
  EtMCJ1  = 0.;
  EtZSP1  = 0.;
  EtJPT1  = 0.;
  DRMAXgjet1 = 1000.;
  ElectronFlagGen1=0;
  MuonFlagGen1=0;
  TauFlagGen1=0;
  ElectronFlagGen1NoLep=0;
  MuonFlagGen1NoLep=0;
  TauFlagGen1NoLep=0;
  BJet1FlagGen=0;
  BJet2FlagGen=0;
  EtaGen2 = 0.;
  PhiGen2 = 0.;
  EtaRaw2 = 0.;
  PhiRaw2 = 0.;
  EtGen2  = 0.;
  EtRaw2  = 0.;
  EtMCJ2  = 0.;
  EtZSP2  = 0.;
  EtJPT2  = 0.;
  DRMAXgjet2 = 1000.;
  ElectronFlagGen2=0;
  MuonFlagGen2=0;
  TauFlagGen2=0;
  ElectronFlagGen2NoLep=0;
  MuonFlagGen2NoLep=0;
  TauFlagGen2NoLep=0;
  hoE=0.;
  nelecs=0;
  nmuons=0;
  ntaus=0;
  for(unsigned i = 0; i < 10; i++){
    elecMom[i]=0;
    muonMom[i]=0;
    tauMom[i]=0;
    elecPt[i]=0;
    muonPt[i]=0;
    tauPt[i]=0;
  }

}
#endif // #ifdef JPTBjetRootAnalysis_cxx
