//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Jul  4 15:05:12 2008 by ROOT version 5.18/00a
// from TTree AnalysisTree/MBUE Analysis Tree 
// found on file: dcap://dcache-ses-cms.desy.de:22125/pnfs/desy.de/cms/tier2/store/user/bechtel/CSA08/S156/MinBias/MBUEAnalysisRootFile_1.root
//////////////////////////////////////////////////////////

#ifndef AntiKtAnalysis_h
#define AntiKtAnalysis_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include <TClonesArray.h>
#include <TLorentzVector.h>

#include <TH1F.h>
#include <TH2D.h>
#include <TProfile.h>

// FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include "fastjet/JetDefinition.hh"

using namespace std;


class AntiKtAnalysis : public TSelector {
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
   Double_t        genEventScale;

   // List of branches
   TBranch        *b_EventKind;   //!
   TBranch        *b_MonteCarlo;   //!
   TBranch        *b_Track;   //!
   TBranch        *b_InclusiveJet;   //!
   TBranch        *b_ChargedJet;   //!
   TBranch        *b_TracksJet;   //!
   TBranch        *b_CalorimeterJet;   //!
   TBranch        *b_acceptedTriggers;   //!
   TBranch        *b_genEventScale;   //!

   AntiKtAnalysis(TTree * /*tree*/ =0) { }
   virtual ~AntiKtAnalysis() { }
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

   virtual double ecalPhi(const float);

   TH1D* h_pTJet;
   TH1D* h_nConstituents;
   TH1D* h_pTSumConstituents;
   TH1D* h_pTByNConstituents;
   TH1D* h_areaJet1;
   TH1D* h_pTConstituent;
   TH1D* h_dphiJC;
   TH1D* h_dphiEcal;
   TH1D* h_pTAllJets;
   TH1D* h_areaAllJets;
   TH1D* h_pTByAreaAllJets;

   TH2D* h2d_nConstituents_vs_pTJet;
   TH2D* h2d_pTSumConstituents_vs_pTJet;
   TH2D* h2d_pTByNConstituents_vs_pTJet;
   TH2D* h2d_areaJet1_vs_pTJet1;
   TH2D* h2d_pTConstituent_vs_pTJet;
   TH2D* h2d_dphiJC_vs_pTConstituent;
   TH2D* h2d_dphiJC_vs_pTJet;
   TH2D* h2d_dphiEcal_vs_pTConstituent;
   TH2D* h2d_dphiEcal_vs_pTJet;
   TH2D* h2d_pTByAreaAllJets_vs_pTJet;

   ClassDef(AntiKtAnalysis,0);
};

#endif

#ifdef AntiKtAnalysis_cxx
void AntiKtAnalysis::Init(TTree *tree)
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
   fChain->SetBranchAddress("genEventScale", &genEventScale, &b_genEventScale);


}

Bool_t AntiKtAnalysis::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

#endif // #ifdef AntiKtAnalysis_cxx
