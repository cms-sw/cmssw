#ifndef UEAnalysisOnRootple_h
#define UEAnalysisOnRootple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1F.h>
#include <TProfile.h>
#include <TH2D.h>
#include <iostream>
#include <fstream>
#include <vector>

//
#include <TClonesArray.h>
#include <TObjString.h>
//


#include "UEAnalysisUE.h"
#include "UEAnalysisJets.h"
#include "UEAnalysisAntiKtJets.h"
#include "UEAnalysisMPI.h"

using namespace std;

class UEAnalysisOnRootple {
public :

  // declare file handle here
  TFile *f;

  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain
  // Declaration of leave types
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
  TBranch        *b_genEventScale;

  UEAnalysisOnRootple();
  virtual ~UEAnalysisOnRootple();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     MultiAnalysis(char* filelist,char* outname,vector<float> weight,Float_t eta,string type,string trigger,string tkpt,Float_t ptCut);
  virtual void     Init(TTree *tree);
  virtual void     BeginJob(char* outname,string);
  virtual void     EndJob(string);
  virtual void     Loop(Float_t we,Float_t triggerPt,string type,string trigger,string tkpt);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

  TFile* hFile;

  Float_t etaRegion;
  Float_t ptThreshold;

  UEAnalysisUE* ueHLTMinBiasPixel;
  UEAnalysisUE* ueHLTMinBiasHcal ;
  UEAnalysisUE* ueHLTMinBiasEcal ;
  UEAnalysisUE* ueHLTMinBias     ;
  UEAnalysisUE* ueHLTZeroBias    ;
  UEAnalysisUE* ueHLT1jet30      ;
  UEAnalysisUE* ueHLT1jet50      ;
  UEAnalysisUE* ueHLT1jet80      ;
  UEAnalysisUE* ueHLT1jet110     ;
  UEAnalysisUE* ueHLT1jet180     ;
  UEAnalysisUE* ueHLT1jet250     ;
  UEAnalysisUE* ueAll            ;

  UEAnalysisJets* jetsHLTMinBiasPixel;
  UEAnalysisJets* jetsHLTMinBiasHcal ;
  UEAnalysisJets* jetsHLTMinBiasEcal ;
  UEAnalysisJets* jetsHLTMinBias     ;
  UEAnalysisJets* jetsHLTZeroBias    ;
  UEAnalysisJets* jetsHLT1jet30      ;
  UEAnalysisJets* jetsHLT1jet50      ;
  UEAnalysisJets* jetsHLT1jet80      ;
  UEAnalysisJets* jetsHLT1jet110     ;
  UEAnalysisJets* jetsHLT1jet180     ;
  UEAnalysisJets* jetsHLT1jet250     ;
  UEAnalysisJets* jetsAll            ;

  UEAnalysisAntiKtJets* antiKtJets;
  UEAnalysisMPI* mpi;


  //
  TH1D* h_acceptedTriggers;
  TH1D* h_eventScale;
  double pThatMax;
  //

};

#endif
