#ifndef JetComparison_H
#define JetComparison_H
 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"


#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

#include <TFile.h>
#include <TH1.h>
#include <TGraphErrors.h>
#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>




class JetComparison : public edm::stream::EDAnalyzer <> {
 public:
   JetComparison(edm::ParameterSet const& conf);
  ~JetComparison();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) ;
  virtual void endJob() ;
  double deltaR2(double eta0, double phi0, double eta, double phi);
 private:
  
  std::string outputFile_;





 TFile * fFile;
 TGraphErrors * gr;

int nEvent;
  TH1F *   meEtJet;
  TH1F *    meEtGen ;
  TH1F *    meEtJetMatched; 
  TH1F *    meEtGenMatched ;
  TH1F *    meEtaJet;
  TH1F *    meEtaGen;
  TH2F *    meRatio;
  TH2F *    meEnergyHcalvsEcal;  
  TH1F *    meEnergyHO;
  TH1F *    meEnergyHcal;
  TH1F *    meEnergyEcal;
  TH1F *    meNumFiredTowers;
  TH1F *    meEnergyEcalTower;
  TH1F *    meEnergyHcalTower;
  TH1F *    meTotEnergy;
  TH1F *   meNumberJet;
  TH1F *  meDistR;
  TH2F *    meDistR_vs_eta;
  TH2F * meHadronicFrac_vs_eta;
  TH2F *  meNTowers90_vs_eta;
  TH2F *  meNTowers60_vs_eta;
  TH2F * meNTowers_vs_eta;
  float fMinEnergy;

};

#endif
