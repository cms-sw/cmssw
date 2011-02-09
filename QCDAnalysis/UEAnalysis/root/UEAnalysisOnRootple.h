//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Sep 11 11:54:39 2007 by ROOT version 5.14/00b
// from TTree UEAnalysisTree/UE Analysis Tree 
// found on file: MB_Pt05/UnderlyingEvent_RootFile_Result.root
//////////////////////////////////////////////////////////

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

//
#include <TClonesArray.h>
#include <TObjString.h>
//

class UEAnalysisOnRootple {
public :

  // declare file handle here
  TFile *f;

  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain
  
  // Declaration of leave types
  Int_t           EventKind;
  Int_t           NumberMCParticles;
  Float_t         MomentumMC[1000];   //[NumberMCParticles]
  Float_t         TransverseMomentumMC[1000];   //[NumberMCParticles]
  Float_t         EtaMC[1000];   //[NumberMCParticles]
  Float_t         PhiMC[1000];   //[NumberMCParticles]
  Int_t           NumberTracks;
  Float_t         MomentumTK[1000];   //[NumberTracks]
  Float_t         TrasverseMomentumTK[1000];   //[NumberTracks]
  Float_t         EtaTK[1000];   //[NumberTracks]
  Float_t         PhiTK[1000];   //[NumberTracks]
  Int_t           NumberInclusiveJet;
  Float_t         MomentumIJ[1000];   //[NumberInclusiveJet]
  Float_t         TrasverseMomentumIJ[1000];   //[NumberInclusiveJet]
  Float_t         EtaIJ[1000];   //[NumberInclusiveJet]
  Float_t         PhiIJ[1000];   //[NumberInclusiveJet]
  Int_t           NumberChargedJet;
  Float_t         MomentumCJ[1000];   //[NumberChargedJet]
  Float_t         TrasverseMomentumCJ[1000];   //[NumberChargedJet]
  Float_t         EtaCJ[1000];   //[NumberChargedJet]
  Float_t         PhiCJ[1000];   //[NumberChargedJet]
  Int_t           NumberTracksJet;
  Float_t         MomentumTJ[1000];   //[NumberTracksJet]
  Float_t         TrasverseMomentumTJ[1000];   //[NumberTracksJet]
  Float_t         EtaTJ[1000];   //[NumberTracksJet]
  Float_t         PhiTJ[1000];   //[NumberTracksJet]
  Int_t           NumberCaloJet;
  Float_t         MomentumEHJ[1000];   //[NumberCaloJet]
  Float_t         TrasverseMomentumEHJ[1000];   //[NumberCaloJet]
  Float_t         EtaEHJ[1000];   //[NumberCaloJet]
  Float_t         PhiEHJ[1000];   //[NumberCaloJet]

  //
  TClonesArray    *acceptedTriggers;
  //

  // List of branches
  TBranch        *b_EventKind;   //!
  TBranch        *b_NumberMCParticles;   //!
  TBranch        *b_MomentumMC;   //!
  TBranch        *b_TransverseMomentumMC;   //!
  TBranch        *b_EtaMC;   //!
  TBranch        *b_PhiMC;   //!
  TBranch        *b_NumberTracks;   //!
  TBranch        *b_MomentumTK;   //!
  TBranch        *b_TrasverseMomentumTK;   //!
  TBranch        *b_EtaTK;   //!
  TBranch        *b_PhiTK;   //!
  TBranch        *b_NumberInclusiveJet;   //!
  TBranch        *b_MomentumIJ;   //!
  TBranch        *b_TrasverseMomentumIJ;   //!
  TBranch        *b_EtaIJ;   //!
  TBranch        *b_PhiIJ;   //!
  TBranch        *b_NumberChargedJet;   //!
  TBranch        *b_MomentumCJ;   //!
  TBranch        *b_TrasverseMomentumCJ;   //!
  TBranch        *b_EtaCJ;   //!
  TBranch        *b_PhiCJ;   //!
  TBranch        *b_NumberTracksJet;   //!
  TBranch        *b_MomentumTJ;   //!
  TBranch        *b_TrasverseMomentumTJ;   //!
  TBranch        *b_EtaTJ;   //!
  TBranch        *b_PhiTJ;   //!
  TBranch        *b_NumberCaloJet;   //!
  TBranch        *b_MomentumEHJ;   //!
  TBranch        *b_TrasverseMomentumEHJ;   //!
  TBranch        *b_EtaEHJ;   //!
  TBranch        *b_PhiEHJ;   //!

  //
  TBranch        *b_acceptedTriggers; 
  //


  //Charged Jet caharacterization
  TH1F* dr_chgcalo;
  TH1F* dr_chginc;
  TH1F* dr_chgmcreco;
  TH1F* dr_caloinc;
  TH1F* numb_cal;
  TH1F* pT_cal;
  TH1F* eta_cal;
  TH1F* eta_cal_res;
  TH1F* phi_cal;
  TH1F* phi_cal_res;
  TH1F* numb_chgmc;
  TH1F* pT_chgmc;
  TH1F* eta_chgmc;
  TH1F* eta_chgmc_res;
  TH1F* phi_chgmc;
  TH1F* phi_chgmc_res;
  TH1F* numb_chgreco;
  TH1F* pT_chgreco;
  TH1F* eta_chgreco;
  TH1F* eta_chgreco_res;
  TH1F* phi_chgreco;
  TH1F* phi_chgreco_res;
  TH1F* numb_inc;
  TH1F* pT_inc;
  TH1F* eta_inc;
  TH1F* phi_inc;
  TProfile* calib_chgcalo;
  TProfile* calib_chginc;
  TProfile* calib_chgmcreco;
  TProfile* calib_caloinc;
  TProfile* calib_chgcalo_eta;
  TProfile* calib_chginc_eta;
  TProfile* calib_chgmcreco_eta;
  TProfile* calib_caloinc_eta;
  TProfile* calib_chgcalo_phi;
  TProfile* calib_chginc_phi;
  TProfile* calib_chgmcreco_phi;
  TProfile* calib_caloinc_phi;

  //Underlying Event analysis
  TH1F*       fHistPtDistMC;
  TH1F*       fHistEtaDistMC;
  TH1F*       fHistPhiDistMC;

  TProfile*   pdN_vs_etaMC;
  TProfile*   pdN_vs_ptMC;

  TProfile*   pdN_vs_dphiMC;
  TProfile*   pdPt_vs_dphiMC;

  // add histo on fluctuation in UE
  TH2D*   h2d_dN_vs_ptJTransMC;


  TProfile*   pdN_vs_ptJTransMC;
  TProfile*   pdN_vs_ptJTransMaxMC;
  TProfile*   pdN_vs_ptJTransMinMC;
  TProfile*   pdPt_vs_ptJTransMC;
  TProfile*   pdPt_vs_ptJTransMaxMC;
  TProfile*   pdPt_vs_ptJTransMinMC;
  TProfile*   pdN_vs_ptJTowardMC;
  TProfile*   pdN_vs_ptJAwayMC;
  TProfile*   pdPt_vs_ptJTowardMC;
  TProfile*   pdPt_vs_ptJAwayMC;

  TH1F*       temp1MC;
  TH1F*       temp2MC;
  TH1F*       temp3MC;
  TH1F*       temp4MC;

  TH1F*       fHistPtDistRECO;
  TH1F*       fHistEtaDistRECO;
  TH1F*       fHistPhiDistRECO;

  TProfile*   pdN_vs_etaRECO;
  TProfile*   pdN_vs_ptRECO;

  TProfile*   pdN_vs_dphiRECO;
  TProfile*   pdPt_vs_dphiRECO;

  TProfile*   pdN_vs_ptJTransRECO;
  TProfile*   pdN_vs_ptJTransMaxRECO;
  TProfile*   pdN_vs_ptJTransMinRECO;
  TProfile*   pdPt_vs_ptJTransRECO;
  TProfile*   pdPt_vs_ptJTransMaxRECO;
  TProfile*   pdPt_vs_ptJTransMinRECO;
  TProfile*   pdN_vs_ptJTowardRECO;
  TProfile*   pdN_vs_ptJAwayRECO;
  TProfile*   pdPt_vs_ptJTowardRECO;
  TProfile*   pdPt_vs_ptJAwayRECO;

  TProfile*   pdN_vs_ptCJTransRECO;
  TProfile*   pdN_vs_ptCJTransMaxRECO;
  TProfile*   pdN_vs_ptCJTransMinRECO;
  TProfile*   pdPt_vs_ptCJTransRECO;
  TProfile*   pdPt_vs_ptCJTransMaxRECO;
  TProfile*   pdPt_vs_ptCJTransMinRECO;
  TProfile*   pdN_vs_ptCJTowardRECO;
  TProfile*   pdN_vs_ptCJAwayRECO;
  TProfile*   pdPt_vs_ptCJTowardRECO;
  TProfile*   pdPt_vs_ptCJAwayRECO;

  TH1F*       temp1RECO;
  TH1F*       temp2RECO;
  TH1F*       temp3RECO;
  TH1F*       temp4RECO;

  TH1D* fNumbMPIMC;
  TH1D* fdEtaLeadingPairMC;
  TH1D* fdPhiLeadingPairMC;
  TH1D* fptRatioLeadingPairMC;
  TProfile* pPtRatio_vs_PtJleadMC;
  TProfile* pPtRatio_vs_EtaJleadMC;
  TProfile* pPtRatio_vs_PhiJleadMC;

  TH1D* fNumbMPIRECO;
  TH1D* fdEtaLeadingPairRECO;
  TH1D* fdPhiLeadingPairRECO;
  TH1D* fptRatioLeadingPairRECO;
  TProfile* pPtRatio_vs_PtJleadRECO;
  TProfile* pPtRatio_vs_EtaJleadRECO;
  TProfile* pPtRatio_vs_PhiJleadRECO;


  
  Float_t etaRegion;
  Float_t piG;
  Float_t rangePhi;
  Float_t ptThreshold;
  
  UEAnalysisOnRootple();
  virtual ~UEAnalysisOnRootple();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     MultiAnalysis(char* filelist,char* outname,Float_t weight[13],Float_t eta,Float_t triggerPt,std::string type,std::string trigger,std::string tkpt,Float_t ptCut);
  virtual void     Init(TTree *tree);
  virtual void     BeginJob(char* outname);
  virtual void     EndJob();
  virtual void     Loop(Float_t we,Float_t triggerPt,std::string type,std::string trigger,std::string tkpt);
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  virtual void     UEAnalysisMC(Float_t weight,std::string tkpt);
  virtual void     UEAnalysisRECO(Float_t weight,std::string tkpt);
  virtual void     JetCalibAnalysis(Float_t weight,std::string tkpt);
  virtual void     MPIAnalysisMC(Float_t weight,std::string tkpt);
  virtual void     MPIAnalysisRECO(Float_t weight,std::string tkpt);
  Float_t CalibrationPt(Float_t ptReco,std::string tkpt);
  Float_t CorrectionPtTrans(Float_t ptReco,std::string tkpt);
  Float_t CorrectionPtToward(Float_t ptReco,std::string tkpt);
  Float_t CorrectionPtAway(Float_t ptReco,std::string tkpt);
  Float_t CorrectionNTrans(Float_t ptReco,std::string tkpt);
  Float_t CorrectionNToward(Float_t ptReco,std::string tkpt);
  Float_t CorrectionNAway(Float_t ptReco,std::string tkpt);

  TFile* hFile;

};

#endif

#ifdef UEAnalysisOnRootple_cxx

UEAnalysisOnRootple::UEAnalysisOnRootple()
{
  std::cout << "UEAnalysisOnRootple constructor " <<std::endl;
}

UEAnalysisOnRootple::~UEAnalysisOnRootple()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t UEAnalysisOnRootple::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t UEAnalysisOnRootple::LoadTree(Long64_t entry)
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

void UEAnalysisOnRootple::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normaly not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

  // allocate space for file handle here
  f = new TFile;

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   //
   acceptedTriggers = 0;
   fChain->SetBranchAddress("acceptedTriggers", &acceptedTriggers, &b_acceptedTriggers);
   //

   fChain->SetBranchAddress("EventKind", &EventKind, &b_EventKind);
   fChain->SetBranchAddress("NumberMCParticles", &NumberMCParticles, &b_NumberMCParticles);
   fChain->SetBranchAddress("MomentumMC", MomentumMC, &b_MomentumMC);
   fChain->SetBranchAddress("TransverseMomentumMC", TransverseMomentumMC, &b_TransverseMomentumMC);
   fChain->SetBranchAddress("EtaMC", EtaMC, &b_EtaMC);
   fChain->SetBranchAddress("PhiMC", PhiMC, &b_PhiMC);
   fChain->SetBranchAddress("NumberTracks", &NumberTracks, &b_NumberTracks);
   fChain->SetBranchAddress("MomentumTK", MomentumTK, &b_MomentumTK);
   fChain->SetBranchAddress("TrasverseMomentumTK", TrasverseMomentumTK, &b_TrasverseMomentumTK);
   fChain->SetBranchAddress("EtaTK", EtaTK, &b_EtaTK);
   fChain->SetBranchAddress("PhiTK", PhiTK, &b_PhiTK);
   fChain->SetBranchAddress("NumberInclusiveJet", &NumberInclusiveJet, &b_NumberInclusiveJet);
   fChain->SetBranchAddress("MomentumIJ", MomentumIJ, &b_MomentumIJ);
   fChain->SetBranchAddress("TrasverseMomentumIJ", TrasverseMomentumIJ, &b_TrasverseMomentumIJ);
   fChain->SetBranchAddress("EtaIJ", EtaIJ, &b_EtaIJ);
   fChain->SetBranchAddress("PhiIJ", PhiIJ, &b_PhiIJ);
   fChain->SetBranchAddress("NumberChargedJet", &NumberChargedJet, &b_NumberChargedJet);
   fChain->SetBranchAddress("MomentumCJ", MomentumCJ, &b_MomentumCJ);
   fChain->SetBranchAddress("TrasverseMomentumCJ", TrasverseMomentumCJ, &b_TrasverseMomentumCJ);
   fChain->SetBranchAddress("EtaCJ", EtaCJ, &b_EtaCJ);
   fChain->SetBranchAddress("PhiCJ", PhiCJ, &b_PhiCJ);
   fChain->SetBranchAddress("NumberTracksJet", &NumberTracksJet, &b_NumberTracksJet);
   fChain->SetBranchAddress("MomentumTJ", MomentumTJ, &b_MomentumTJ);
   fChain->SetBranchAddress("TrasverseMomentumTJ", TrasverseMomentumTJ, &b_TrasverseMomentumTJ);
   fChain->SetBranchAddress("EtaTJ", EtaTJ, &b_EtaTJ);
   fChain->SetBranchAddress("PhiTJ", PhiTJ, &b_PhiTJ);
   fChain->SetBranchAddress("NumberCaloJet", &NumberCaloJet, &b_NumberCaloJet);
   fChain->SetBranchAddress("MomentumEHJ", MomentumEHJ, &b_MomentumEHJ);
   fChain->SetBranchAddress("TrasverseMomentumEHJ", TrasverseMomentumEHJ, &b_TrasverseMomentumEHJ);
   fChain->SetBranchAddress("EtaEHJ", EtaEHJ, &b_EtaEHJ);
   fChain->SetBranchAddress("PhiEHJ", PhiEHJ, &b_PhiEHJ);
   Notify();
}

Bool_t UEAnalysisOnRootple::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void UEAnalysisOnRootple::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

Int_t UEAnalysisOnRootple::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif
