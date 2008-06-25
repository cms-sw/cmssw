#ifndef UEAnalysisAntiKtJets_h
#define UEAnalysisAntiKtJets_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include<TLorentzVector.h>

#include <TH1F.h>
#include <TH2D.h>
#include <TProfile.h>

#include <TClonesArray.h>

// FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include "fastjet/JetDefinition.hh"


using namespace std;

class UEAnalysisAntiKtJets {
 public :

  UEAnalysisAntiKtJets();
  ~UEAnalysisAntiKtJets(){}
  double ecalPhi(const float);
  void   jetAnalysis(float, float, float, TClonesArray*, TFile* , string);
  //  void jetCalibAnalysis(float ,float,TClonesArray *,TClonesArray *,TClonesArray *,TClonesArray *, TClonesArray *,TClonesArray *, TClonesArray* );
  void writeToFile(TFile *);

  void Begin(TFile *, string );

  TH1D* h_pTJet;
  TH1D* h_nConstituentsJet;
  TH1D* h_pTByNConstituentsJet;
  TH1D* h_areaJet;
  TH1D* h_pTConstituent;
  TH1D* h_dphiJC;
  TH1D* h_dphiEcal;

  TH2D* h2d_nConstituentsJet_vs_pTJet;
  TH2D* h2d_pTByNConstituentsJet_vs_pTJet;
  TH2D* h2d_areaJet_vs_pTJet;
  TH2D* h2d_pTConstituent_vs_pTJet;
  TH2D* h2d_dphiJC_vs_pTConstituent;
  TH2D* h2d_dphiJC_vs_pTJet;
  TH2D* h2d_dphiEcal_vs_pTConstituent;
  TH2D* h2d_dphiEcal_vs_pTJet;

  float piG;
};

#endif
