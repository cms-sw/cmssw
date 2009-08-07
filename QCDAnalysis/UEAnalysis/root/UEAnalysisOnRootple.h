#ifndef UEAnalysisOnRootple_h
#define UEAnalysisOnRootple_h

#include <vector>
#include <fstream>
#include <iostream>

#include <TH1F.h>
#include <TH2D.h>
#include <TROOT.h>
#include <TFile.h>
#include <TChain.h>
#include <TProfile.h>
#include <TObjString.h>
#include <TClonesArray.h>

#include "UETrigger.h"
#include "UEJetArea.h"
#include "UEActivity.h"
#include "UEAnalysisUE.h"
#include "UEAnalysisMPI.h"
#include "UEAnalysisJets.h"
#include "UEAnalysisGAM.h"

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
  TClonesArray    *MCGamma;
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
  TBranch        *b_MCGamma;
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
  //  virtual void     Init(TTree *tree);
  virtual void     Init(TTree *tree, string);
  //  UEAnalysisOnRootple.C:169: error: no matching function for call to `UEAnalysisOnRootple::Init(TTree*&, std::string&)'
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

  UEAnalysisMPI* mpi;
  UEAnalysisGAM* gam;


  TH1D* h_acceptedTriggers;
  TH1D* h_eventScale;
  double pThatMax;

  std::string HLTBitNames[11]; 

  string SampleType;
  UEJetAreaHistograms*  areaHistos;

  //  UEActivityFinder*     activityFinder;
  UEActivityHistograms* ueHistos;
  UETriggerHistograms*  hltHistos;
};

#endif
