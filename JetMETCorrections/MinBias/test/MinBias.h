//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Jul 16 10:53:19 2006 by ROOT version 5.11/06a
// from TTree RecJet/RecJet Tree
// found on file: analysis.root
//////////////////////////////////////////////////////////

#ifndef MinBias_h
#define MinBias_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

class MinBias {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leave types
   Int_t           mydet;
   Int_t           mysubd;
   Int_t           depth;
   Int_t           ieta;
   Int_t           iphi;
   Float_t         eta;
   Float_t         phi;
   Float_t         mom1;
   Float_t         mom2;
   Float_t         mom3;
   Float_t         mom4;

   // List of branches
   TBranch        *b_mydet;   //!
   TBranch        *b_mysubd;   //!
   TBranch        *b_depth;   //!
   TBranch        *b_ieta;   //!
   TBranch        *b_iphi;   //!
   TBranch        *b_eta;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_mom1;   //!
   TBranch        *b_mom2;   //!
   TBranch        *b_mom3;   //!
   TBranch        *b_mom4;   //!

   MinBias(TTree *tree=0);
   virtual ~MinBias();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef MinBias_cxx
MinBias::MinBias(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("analysis.root");
      if (!f) {
         f = new TFile("analysis.root");
      }
      tree = (TTree*)gDirectory->Get("RecJet");

   }
   Init(tree);
}

MinBias::~MinBias()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MinBias::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MinBias::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->IsA() != TChain::Class()) return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void MinBias::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses of the tree
   // will be set. It is normaly not necessary to make changes to the
   // generated code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running with PROOF.

   // Set branch addresses
   if (tree == 0) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("mydet",&mydet);
   fChain->SetBranchAddress("mysubd",&mysubd);
   fChain->SetBranchAddress("depth",&depth);
   fChain->SetBranchAddress("ieta",&ieta);
   fChain->SetBranchAddress("iphi",&iphi);
   fChain->SetBranchAddress("eta",&eta);
   fChain->SetBranchAddress("phi",&phi);
   fChain->SetBranchAddress("mom1",&mom1);
   fChain->SetBranchAddress("mom2",&mom2);
   fChain->SetBranchAddress("mom3",&mom3);
   fChain->SetBranchAddress("mom4",&mom4);
   Notify();
}

Bool_t MinBias::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   b_mydet = fChain->GetBranch("mydet");
   b_mysubd = fChain->GetBranch("mysubd");
   b_depth = fChain->GetBranch("depth");
   b_ieta = fChain->GetBranch("ieta");
   b_iphi = fChain->GetBranch("iphi");
   b_eta = fChain->GetBranch("eta");
   b_phi = fChain->GetBranch("phi");
   b_mom1 = fChain->GetBranch("mom1");
   b_mom2 = fChain->GetBranch("mom2");
   b_mom3 = fChain->GetBranch("mom3");
   b_mom4 = fChain->GetBranch("mom4");

   return kTRUE;
}

void MinBias::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t MinBias::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef MinBias_cxx
