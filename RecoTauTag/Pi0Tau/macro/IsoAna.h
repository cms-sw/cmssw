//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Aug  4 18:34:30 2008 by ROOT version 5.18/00a
// from TTree tree/tree
// found on file: hist_stau.root
//////////////////////////////////////////////////////////

#ifndef IsoAna_h
#define IsoAna_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <TString.h>

class IsoAna {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Int_t           nSignalTracks;
   Int_t           nSignalPi0s;
   Int_t           nIsolationTracks;
   Int_t           nIsolationPi0s;
   TLorentzVector  *tracksMomentum;
   TLorentzVector  *pi0sMomentum;
   TLorentzVector  *momentum;

   // List of branches
   TBranch        *b_nSignalTracks;   //!
   TBranch        *b_nSignalPi0s;   //!
   TBranch        *b_nIsolationTracks;   //!
   TBranch        *b_nIsolationPi0s;   //!
   TBranch        *b_tracksMomentum;
   TBranch        *b_pi0sMomentum;
   TBranch        *b_momentum;

   IsoAna(TTree *tree=0);
   virtual ~IsoAna();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(TString ds = "stau");
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef IsoAna_cxx
IsoAna::IsoAna(TTree *tree)
{

  tracksMomentum = new TLorentzVector;
  pi0sMomentum = new TLorentzVector;
  momentum = new TLorentzVector;

// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("hist_stau.root");
      if (!f) {
         f = new TFile("hist_stau.root");
      }
      tree = (TTree*)gDirectory->Get("tree");

   }
   Init(tree);
}

IsoAna::~IsoAna()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t IsoAna::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t IsoAna::LoadTree(Long64_t entry)
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

void IsoAna::Init(TTree *tree)
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
   //   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("nSignalTracks", &nSignalTracks, &b_nSignalTracks);
   fChain->SetBranchAddress("nSignalPi0s", &nSignalPi0s, &b_nSignalPi0s);
   fChain->SetBranchAddress("nIsolationTracks", &nIsolationTracks, &b_nIsolationTracks);
   fChain->SetBranchAddress("nIsolationPi0s", &nIsolationPi0s, &b_nIsolationPi0s);
   fChain->SetBranchAddress("tracksMomentum.", &tracksMomentum, &b_tracksMomentum);
   fChain->SetBranchAddress("pi0sMomentum.", &pi0sMomentum, &b_pi0sMomentum);
   fChain->SetBranchAddress("momentum.", &momentum, &b_momentum);
   Notify();
}

Bool_t IsoAna::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void IsoAna::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t IsoAna::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef IsoAna_cxx
