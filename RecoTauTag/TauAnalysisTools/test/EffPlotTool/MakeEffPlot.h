//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Apr 29 14:53:10 2013 by ROOT version 5.32/00
// from TTree Ntuple/Expression Ntuple
// found on file: TriggerEfficiencyTree.root
//////////////////////////////////////////////////////////

#ifndef MakeEffPlot_h
#define MakeEffPlot_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

// Fixed size dimensions of array or collections stored in the TTree if any.

class MakeEffPlot {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Float_t         byLooseCombinedIsolationDeltaBetaCorr;
   Float_t         byMediumCombinedIsolationDeltaBetaCorr;
   Float_t         byTightCombinedIsolationDeltaBetaCorr;
   Float_t         hltPFTau35;
   Float_t         hltPFTau35Track;
   Float_t         hltPFTau35TrackPt20;
   Float_t         index;
   Float_t         nRecoObjects;
   Float_t         tagTauDecayMode;
   Float_t         tagTauEta;
   Float_t         tagTauPhi;
   Float_t         tagTauPt;
   Int_t           idx;

   // List of branches
   TBranch        *b_byLooseCombinedIsolationDeltaBetaCorr;   //!
   TBranch        *b_byMediumCombinedIsolationDeltaBetaCorr;   //!
   TBranch        *b_byTightCombinedIsolationDeltaBetaCorr;   //!
   TBranch        *b_hltPFTau35;   //!
   TBranch        *b_hltPFTau35Track;   //!
   TBranch        *b_hltPFTau35TrackPt20;   //!
   TBranch        *b_index;   //!
   TBranch        *b_nRecoObjects;   //!
   TBranch        *b_tagTauDecayMode;   //!
   TBranch        *b_tagTauEta;   //!
   TBranch        *b_tagTauPhi;   //!
   TBranch        *b_tagTauPt;   //!
   TBranch        *b_idx;   //!

   MakeEffPlot(TTree *tree=0);
   virtual ~MakeEffPlot();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef MakeEffPlot_cxx
MakeEffPlot::MakeEffPlot(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("TriggerEfficiencyTree.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("TriggerEfficiencyTree.root");
      }
      TDirectory * dir = (TDirectory*)f->Get("TriggerEfficiencyTree.root:/triggerMatch");
      dir->GetObject("Ntuple",tree);

   }
   Init(tree);
   this->Loop();
}

MakeEffPlot::~MakeEffPlot()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MakeEffPlot::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MakeEffPlot::LoadTree(Long64_t entry)
{
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

void MakeEffPlot::Init(TTree *tree)
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

   fChain->SetBranchAddress("byLooseCombinedIsolationDeltaBetaCorr", &byLooseCombinedIsolationDeltaBetaCorr, &b_byLooseCombinedIsolationDeltaBetaCorr);
   fChain->SetBranchAddress("byMediumCombinedIsolationDeltaBetaCorr", &byMediumCombinedIsolationDeltaBetaCorr, &b_byMediumCombinedIsolationDeltaBetaCorr);
   fChain->SetBranchAddress("byTightCombinedIsolationDeltaBetaCorr", &byTightCombinedIsolationDeltaBetaCorr, &b_byTightCombinedIsolationDeltaBetaCorr);
   fChain->SetBranchAddress("hltPFTau35", &hltPFTau35, &b_hltPFTau35);
   fChain->SetBranchAddress("hltPFTau35Track", &hltPFTau35Track, &b_hltPFTau35Track);
   fChain->SetBranchAddress("hltPFTau35TrackPt20", &hltPFTau35TrackPt20, &b_hltPFTau35TrackPt20);
   fChain->SetBranchAddress("index", &index, &b_index);
   fChain->SetBranchAddress("nRecoObjects", &nRecoObjects, &b_nRecoObjects);
   fChain->SetBranchAddress("tagTauDecayMode", &tagTauDecayMode, &b_tagTauDecayMode);
   fChain->SetBranchAddress("tagTauEta", &tagTauEta, &b_tagTauEta);
   fChain->SetBranchAddress("tagTauPhi", &tagTauPhi, &b_tagTauPhi);
   fChain->SetBranchAddress("tagTauPt", &tagTauPt, &b_tagTauPt);
   fChain->SetBranchAddress("idx", &idx, &b_idx);
   Notify();
}

Bool_t MakeEffPlot::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void MakeEffPlot::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t MakeEffPlot::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef MakeEffPlot_cxx
