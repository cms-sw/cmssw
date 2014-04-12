#ifndef UEAnalysisMPI_h
#define UEAnalysisMPI_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include <TH1F.h>
#include <TProfile.h>

#include <TClonesArray.h>
#include <TLorentzVector.h>

class UEAnalysisMPI {
 public :

  UEAnalysisMPI();
  ~UEAnalysisMPI(){}

  void mpiAnalysisMC(float,float,float,TClonesArray*);
  void mpiAnalysisRECO(float,float,float,TClonesArray*);

  void Begin(TFile *);

  void writeToFile(TFile *);

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

  float piG;
  float rangePhi;
};

#endif
