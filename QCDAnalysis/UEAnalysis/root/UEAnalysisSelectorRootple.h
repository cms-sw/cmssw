//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Apr 21 14:19:27 2008 by ROOT version 5.18/00
// from TTree AnalysisTree/MBUE Analysis Tree 
// found on file: MBUEAnalysisRootFile.root
//////////////////////////////////////////////////////////

#ifndef UEAnalysisSelectorRootple_h
#define UEAnalysisSelectorRootple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>


#include "UEAnalysisUE.h"
#include "UEAnalysisJets.h"
#include "UEAnalysisMPI.h"

//
#include <TClonesArray.h>
#include <TObjString.h>
//

class UEAnalysisSelectorRootple : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   // Declaration of leaf types
   Int_t           EventKind;
   TClonesArray    *MonteCarlo;
   TClonesArray    *Track;
   TClonesArray    *InclusiveJet;
   TClonesArray    *ChargedJet;
   TClonesArray    *TracksJet;
   TClonesArray    *CalorimeterJet;
   TClonesArray    *acceptedTriggers;

   // List of branches
   TBranch        *b_EventKind;   //!
   TBranch        *b_MonteCarlo;   //!
   TBranch        *b_Track;   //!
   TBranch        *b_InclusiveJet;   //!
   TBranch        *b_ChargedJet;   //!
   TBranch        *b_TracksJet;   //!
   TBranch        *b_CalorimeterJet;   //!
   TBranch        *b_acceptedTriggers;   //!

   UEAnalysisSelectorRootple(TTree * /*tree*/ =0) { }
   virtual ~UEAnalysisSelectorRootple() { }
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(UEAnalysisSelectorRootple,0);

   TFile* hFile;
   
   float etaRegion;
   float ptThreshold;
   
   UEAnalysisUE * ue;
   UEAnalysisJets* jets;
   UEAnalysisMPI* mpi;
   
};

#endif

#ifdef UEAnalysisSelectorRootple_cxx
void UEAnalysisSelectorRootple::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   MonteCarlo = 0;
   Track = 0;
   InclusiveJet = 0;
   ChargedJet = 0;
   TracksJet = 0;
   CalorimeterJet = 0;
   acceptedTriggers = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("EventKind", &EventKind, &b_EventKind);
   fChain->SetBranchAddress("MonteCarlo", &MonteCarlo, &b_MonteCarlo);
   fChain->SetBranchAddress("Track", &Track, &b_Track);
   fChain->SetBranchAddress("InclusiveJet", &InclusiveJet, &b_InclusiveJet);
   fChain->SetBranchAddress("ChargedJet", &ChargedJet, &b_ChargedJet);
   fChain->SetBranchAddress("TracksJet", &TracksJet, &b_TracksJet);
   fChain->SetBranchAddress("CalorimeterJet", &CalorimeterJet, &b_CalorimeterJet);
   fChain->SetBranchAddress("acceptedTriggers", &acceptedTriggers, &b_acceptedTriggers);

}

Bool_t UEAnalysisSelectorRootple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef UEAnalysisSelectorRootple_cxx
