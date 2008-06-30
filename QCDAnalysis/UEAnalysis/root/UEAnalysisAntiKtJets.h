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

  float piG;
};

#endif
