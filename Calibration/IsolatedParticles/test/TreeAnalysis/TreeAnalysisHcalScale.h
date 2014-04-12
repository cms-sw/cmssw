#ifndef TreeAnalysisHcalScale_h
#define TreeAnalysisHcalScale_h

//////////////////////////////////////////////////////////
// This class has been automatically generated and then 
// modified to accept six different files 
//////////////////////////////////////////////////////////

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>


#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TDirectory.h"
#include "TH1F.h"
#include "TH2.h"
#include "TProfile.h"
#include "TChain.h"
#include "TString.h"

class TreeAnalysisHcalScale {

public :
  static const int NEtaBins = 51, NPBins = 5, NParticle=6;
  double genPartPBins[NPBins+1];
  double weights[NParticle];
  //  double genPartEtaBins[4];
  TFile *fout;

  TH1F  *h_trackP[NParticle], *h_trackEtaAll[NParticle], *h_trackPt[NParticle];
  TH1F  *h_trackEta[NParticle], *h_trackPhi[NParticle], *h_IsotrackPhi[NParticle];
  TH1F  *h_trackHcalEta[NParticle], *h_IsotrackHcalIEta[NParticle], *h_FracIsotrackHcalIEta[NParticle], *h_trackHcalPhi[NParticle];
  TH1F  *h_hCone[NParticle], *h_conehmaxNearP[NParticle];
  TH1F  *h_eMipDR[NParticle], *h_eECALDR[NParticle], *h_eHCALDR[NParticle];
  TH1F  *h_e11x11_20Sig[NParticle], *h_e15x15_20Sig[NParticle];
  
  TH1F  *h_eHcalFrac_all[NPBins][NEtaBins];
  TH1F  *h_eHcalFrac_trunc_all[NPBins][NEtaBins];
  TH1F  *h_Response_all[NPBins][NEtaBins];
  TH1F  *h_Response_trunc_all[NPBins][NEtaBins];
  TH1F  *h_Response_E11x11_all[NPBins][NEtaBins];
  TH1F  *h_Response_E11x11_trunc_all[NPBins][NEtaBins];
  
  TH1F  *h_eHcalFrac[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_Response[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_Response_E11x11[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_eHcalFrac_trunc[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_Response_trunc[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_Response_E11x11_trunc[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_hneutIso[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_eneutIso[NParticle+1][NPBins][NEtaBins];
  TH1F  *h_eneutIsoNxN[NParticle+1][NPBins][NEtaBins];

  //================================
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  //   Int_t           t_EvtNo;
  Int_t                 t_RunNo;
  Int_t                 t_Lumi;
  Int_t                 t_Bunch;
  std::vector<double>  *t_trackP;
  std::vector<double>  *t_trackPt;
  std::vector<double>  *t_trackEta;
  std::vector<double>  *t_trackPhi;
  std::vector<double>  *t_trackHcalEta;
  std::vector<double>  *t_trackHcalPhi;
  std::vector<double>  *t_hCone;
  std::vector<double>  *t_conehmaxNearP;
  std::vector<double>  *t_eMipDR;
  std::vector<double>  *t_eMipDR_2;
  std::vector<double>  *t_eECALDR;
  std::vector<double>  *t_eECALDR_2;
  std::vector<double>  *t_eHCALDR;
  std::vector<double>  *t_e11x11_20Sig;
  std::vector<double>  *t_e15x15_20Sig;  
  Int_t                 t_nTracks;

  // List of branches
  //   TBranch        *b_t_EvtNo;   //!
  TBranch        *b_t_RunNo;   //!
  TBranch        *b_t_Lumi;   //!
  TBranch        *b_t_Bunch;   //!
  TBranch        *b_t_trackP;   //!
  TBranch        *b_t_trackPt;   //!
  TBranch        *b_t_trackEta;   //!
  TBranch        *b_t_trackPhi;   //!
  TBranch        *b_t_trackHcalEta;   //!
  TBranch        *b_t_trackHcalPhi;   //!
  TBranch        *b_t_hCone;   //!
  TBranch        *b_t_conehmaxNearP;   //!
  TBranch        *b_t_eMipDR;   //!
  TBranch        *b_t_eMipDR_2;   //!
  TBranch        *b_t_eECALDR;   //
  TBranch        *b_t_eECALDR_2;   //!
  TBranch        *b_t_eHCALDR;   //!
  TBranch        *b_t_e11x11_20Sig;   //!
  TBranch        *b_t_e15x15_20Sig;   //!
  
  unsigned int   ipBin, nmaxBin, nIsoTrkTotal;

  TreeAnalysisHcalScale(const char *outFileName, std::vector<std::string>& particles);
  virtual ~TreeAnalysisHcalScale();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TChain *tree);
  virtual void     Loop(int cut=1 );
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);
  
  void             BookHistograms(const char *outFileName, 
				  std::vector<std::string>& particles);
  void             clear();
  void             setParticle(unsigned int ip, unsigned int nmax);
  void             AddWeight(std::vector<std::string> particleNames);
};

#endif
