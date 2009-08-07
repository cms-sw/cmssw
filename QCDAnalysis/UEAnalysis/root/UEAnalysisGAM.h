#ifndef UEAnalysisGAM_h
#define UEAnalysisGAM_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include <TH1F.h>
#include <TProfile.h>

#include <TClonesArray.h>
#include <TLorentzVector.h>

using namespace std;

class UEAnalysisGAM {
 public :

  UEAnalysisGAM();
  ~UEAnalysisGAM(){}
  
  void gammaAnalysisMC(Float_t,Float_t,Float_t,TClonesArray&,TClonesArray&);
 
  void Begin(TFile *);

  void writeToFile(TFile *);
  
  TH1D* fdPhiGamma1JetMC;
  TH1D* fdPhiGamma2JetMC;
  TH1D* fdPhiGamma3JetMC;
  TH1D* fPtLeadingGammaMC;
  TH1D* fEtaLeadingGammaMC;
  TH1D* fPhiLeadingGammaMC;
  TH1D* fNumbMPIMC;
  TH1D* fdEtaLeadingPairMC;
  TH1D* fdPhiLeadingPairMC;
  TH1D* fptRatioLeadingPairMC;
  TProfile* pPtRatio_vs_PtJleadMC;
  TProfile* pPtRatio_vs_EtaJleadMC;
  TProfile* pPtRatio_vs_PhiJleadMC;
  
  float piG;
  float rangePhi;
};

#endif
