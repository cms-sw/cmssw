//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Jun  2 11:42:20 2006 by ROOT version 4.04/02g
// from TTree theTree/Detector units positions
// found on file: aligned.root
//////////////////////////////////////////////////////////

#ifndef theTree_h
#define theTree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

class theTree {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leave types
   Float_t         x;
   Float_t         y;
   Float_t         z;
   Float_t         phi;
   Float_t         theta;
   Float_t         length;
   Float_t         width;
   Float_t         thick;

   // List of branches
   TBranch        *b_x;   //!
   TBranch        *b_y;   //!
   TBranch        *b_z;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_theta;   //!
   TBranch        *b_length;   //!
   TBranch        *b_width;   //!
   TBranch        *b_thick;   //!

   theTree(TTree *tree=0);
   virtual ~theTree();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef theTree_cxx
theTree::theTree(TTree *tree)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("aligned.root");
      if (!f) {
         f = new TFile("aligned.root");
      }
      tree = (TTree*)gDirectory->Get("theTree");

   }
   Init(tree);
}

theTree::~theTree()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t theTree::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t theTree::LoadTree(Long64_t entry)
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

void theTree::Init(TTree *tree)
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

   fChain->SetBranchAddress("x",&x);
   fChain->SetBranchAddress("y",&y);
   fChain->SetBranchAddress("z",&z);
   fChain->SetBranchAddress("phi",&phi);
   fChain->SetBranchAddress("theta",&theta);
   fChain->SetBranchAddress("length",&length);
   fChain->SetBranchAddress("width",&width);
   fChain->SetBranchAddress("thick",&thick);
   Notify();
}

Bool_t theTree::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   b_x = fChain->GetBranch("x");
   b_y = fChain->GetBranch("y");
   b_z = fChain->GetBranch("z");
   b_phi = fChain->GetBranch("phi");
   b_theta = fChain->GetBranch("theta");
   b_length = fChain->GetBranch("length");
   b_width = fChain->GetBranch("width");
   b_thick = fChain->GetBranch("thick");

   return kTRUE;
}

void theTree::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t theTree::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef theTree_cxx
