//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Nov 17 14:34:20 2008 by ROOT version 5.18/00a
// from TTree t1/analysis tree
// found on file: SinglePionEfficiencyNew.root
//////////////////////////////////////////////////////////

#ifndef SinglePionEfficiencyNew_h
#define SinglePionEfficiencyNew_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

class SinglePionEfficiencyNew {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Double_t        ptSim1;
   Double_t        etaSim1;
   Double_t        phiSim1;
   Double_t        ptSim2;
   Double_t        etaSim2;
   Double_t        phiSim2;
   Double_t        ptTrk1;
   Double_t        etaTrk1;
   Double_t        phiTrk1;
   Double_t        drTrk1;
   Double_t        purityTrk1;
   Double_t        ptTrk2;
   Double_t        etaTrk2;
   Double_t        phiTrk2;
   Double_t        drTrk2;
   Double_t        purityTrk2;
   Double_t        ptPxl1;
   Double_t        etaPxl1;
   Double_t        phiPxl1;
   Double_t        drPxl1;
   Double_t        purityPxl1;
   Double_t        ptPxl2;
   Double_t        etaPxl2;
   Double_t        phiPxl2;
   Double_t        drPxl2;
   Double_t        purityPxl2;
   Double_t        etCalo1;
   Double_t        etCalo2;
   Double_t        eCalo1;
   Double_t        eCalo2;
   Double_t        e1ECAL7x7;
   Double_t        e1ECAL11x11;
   Double_t        e2ECAL7x7;
   Double_t        e2ECAL11x11;
   Double_t        e1HCAL3x3;
   Double_t        e1HCAL5x5;
   Double_t        e2HCAL3x3;
   Double_t        e2HCAL5x5;
   Int_t           trkQuality1;
   Int_t           trkQuality2;
   Int_t           trkNVhits1;
   Int_t           trkNVhits2;
   Int_t           idmax1;
   Int_t           idmax2;

   // List of branches
   TBranch        *b_ptSim1;   //!
   TBranch        *b_etaSim1;   //!
   TBranch        *b_phiSim1;   //!
   TBranch        *b_ptSim2;   //!
   TBranch        *b_etaSim2;   //!
   TBranch        *b_phiSim2;   //!
   TBranch        *b_ptTrk1;   //!
   TBranch        *b_etaTrk1;   //!
   TBranch        *b_phiTrk1;   //!
   TBranch        *b_drTrk1;   //!
   TBranch        *b_purityTrk1;   //!
   TBranch        *b_ptTrk2;   //!
   TBranch        *b_etaTrk2;   //!
   TBranch        *b_phiTrk2;   //!
   TBranch        *b_drTrk2;   //!
   TBranch        *b_purityTrk2;   //!
   TBranch        *b_ptPxl1;   //!
   TBranch        *b_etaPxl1;   //!
   TBranch        *b_phiPxl1;   //!
   TBranch        *b_drPxl1;   //!
   TBranch        *b_purityPxl1;   //!
   TBranch        *b_ptPxl2;   //!
   TBranch        *b_etaPxl2;   //!
   TBranch        *b_phiPxl2;   //!
   TBranch        *b_drPxl2;   //!
   TBranch        *b_purityPxl2;   //!
   TBranch        *b_etCalo1;   //!
   TBranch        *b_etCalo2;   //!
   TBranch        *b_eCalo1;   //!
   TBranch        *b_eCalo2;   //!
   TBranch        *b_e1ECAL7x7;   //!
   TBranch        *b_e1ECAL11x11;   //!
   TBranch        *b_e2ECAL7x7;   //!
   TBranch        *b_e2ECAL11x11;   //!
   TBranch        *b_e1HCAL3x3;   //!
   TBranch        *b_e1HCAL5x5;   //!
   TBranch        *b_e2HCAL3x3;   //!
   TBranch        *b_e2HCAL5x5;   //!
   TBranch        *b_trkQuality1;   //!
   TBranch        *b_trkQuality2;   //!
   TBranch        *b_trkNVhits1;   //!
   TBranch        *b_trkNVhits2;   //!
   TBranch        *b_idmax1;   //!
   TBranch        *b_idmax2;   //!

   SinglePionEfficiencyNew(TTree *tree=0);
   virtual ~SinglePionEfficiencyNew();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef SinglePionEfficiencyNew_cxx
SinglePionEfficiencyNew::SinglePionEfficiencyNew(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("SinglePionEfficiencyNew.root");
      if (!f) {
         f = new TFile("SinglePionEfficiencyNew.root");
      }
      tree = (TTree*)gDirectory->Get("t1");

   }
   Init(tree);
}

SinglePionEfficiencyNew::~SinglePionEfficiencyNew()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t SinglePionEfficiencyNew::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t SinglePionEfficiencyNew::LoadTree(Long64_t entry)
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

void SinglePionEfficiencyNew::Init(TTree *tree)
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

   fChain->SetBranchAddress("ptSim1", &ptSim1, &b_ptSim1);
   fChain->SetBranchAddress("etaSim1", &etaSim1, &b_etaSim1);
   fChain->SetBranchAddress("phiSim1", &phiSim1, &b_phiSim1);
   fChain->SetBranchAddress("ptSim2", &ptSim2, &b_ptSim2);
   fChain->SetBranchAddress("etaSim2", &etaSim2, &b_etaSim2);
   fChain->SetBranchAddress("phiSim2", &phiSim2, &b_phiSim2);
   fChain->SetBranchAddress("ptTrk1", &ptTrk1, &b_ptTrk1);
   fChain->SetBranchAddress("etaTrk1", &etaTrk1, &b_etaTrk1);
   fChain->SetBranchAddress("phiTrk1", &phiTrk1, &b_phiTrk1);
   fChain->SetBranchAddress("drTrk1", &drTrk1, &b_drTrk1);
   fChain->SetBranchAddress("purityTrk1", &purityTrk1, &b_purityTrk1);
   fChain->SetBranchAddress("ptTrk2", &ptTrk2, &b_ptTrk2);
   fChain->SetBranchAddress("etaTrk2", &etaTrk2, &b_etaTrk2);
   fChain->SetBranchAddress("phiTrk2", &phiTrk2, &b_phiTrk2);
   fChain->SetBranchAddress("drTrk2", &drTrk2, &b_drTrk2);
   fChain->SetBranchAddress("purityTrk2", &purityTrk2, &b_purityTrk2);
   fChain->SetBranchAddress("ptPxl1", &ptPxl1, &b_ptPxl1);
   fChain->SetBranchAddress("etaPxl1", &etaPxl1, &b_etaPxl1);
   fChain->SetBranchAddress("phiPxl1", &phiPxl1, &b_phiPxl1);
   fChain->SetBranchAddress("drPxl1", &drPxl1, &b_drPxl1);
   fChain->SetBranchAddress("purityPxl1", &purityPxl1, &b_purityPxl1);
   fChain->SetBranchAddress("ptPxl2", &ptPxl2, &b_ptPxl2);
   fChain->SetBranchAddress("etaPxl2", &etaPxl2, &b_etaPxl2);
   fChain->SetBranchAddress("phiPxl2", &phiPxl2, &b_phiPxl2);
   fChain->SetBranchAddress("drPxl2", &drPxl2, &b_drPxl2);
   fChain->SetBranchAddress("purityPxl2", &purityPxl2, &b_purityPxl2);
   fChain->SetBranchAddress("etCalo1", &etCalo1, &b_etCalo1);
   fChain->SetBranchAddress("etCalo2", &etCalo2, &b_etCalo2);
   fChain->SetBranchAddress("eCalo1", &eCalo1, &b_eCalo1);
   fChain->SetBranchAddress("eCalo2", &eCalo2, &b_eCalo2);
   fChain->SetBranchAddress("e1ECAL7x7", &e1ECAL7x7, &b_e1ECAL7x7);
   fChain->SetBranchAddress("e1ECAL11x11", &e1ECAL11x11, &b_e1ECAL11x11);
   fChain->SetBranchAddress("e2ECAL7x7", &e2ECAL7x7, &b_e2ECAL7x7);
   fChain->SetBranchAddress("e2ECAL11x11", &e2ECAL11x11, &b_e2ECAL11x11);
   fChain->SetBranchAddress("e1HCAL3x3", &e1HCAL3x3, &b_e1HCAL3x3);
   fChain->SetBranchAddress("e1HCAL5x5", &e1HCAL5x5, &b_e1HCAL5x5);
   fChain->SetBranchAddress("e2HCAL3x3", &e2HCAL3x3, &b_e2HCAL3x3);
   fChain->SetBranchAddress("e2HCAL5x5", &e2HCAL5x5, &b_e2HCAL5x5);
   fChain->SetBranchAddress("trkQuality1", &trkQuality1, &b_trkQuality1);
   fChain->SetBranchAddress("trkQuality2", &trkQuality2, &b_trkQuality2);
   fChain->SetBranchAddress("trkNVhits1", &trkNVhits1, &b_trkNVhits1);
   fChain->SetBranchAddress("trkNVhits2", &trkNVhits2, &b_trkNVhits2);
   fChain->SetBranchAddress("idmax1", &idmax1, &b_idmax1);
   fChain->SetBranchAddress("idmax2", &idmax2, &b_idmax2);
   Notify();
}

Bool_t SinglePionEfficiencyNew::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void SinglePionEfficiencyNew::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t SinglePionEfficiencyNew::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef SinglePionEfficiencyNew_cxx
