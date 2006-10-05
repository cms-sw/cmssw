#ifndef RecoExamples_JetValidation_h
#define RecoExamples_JetValidation_h
#include <TH1.h>
#include <TProfile.h>
#include <TH2.h>
/* \class JetValidation
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class JetValidation : public edm::EDAnalyzer {
public:
  JetValidation( const edm::ParameterSet & );

private:
  //Framwework stuff
  void beginJob( const edm::EventSetup & );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();

  // Parameters passed via the config file
  double PtHistMax;      // Maximum edge of Pt histograms
  int    diagPrintNum;   // Number of events for diagnostic printout
  std::string GenType;   // Type of Generated Particle on Input to analysis
    
  // Root objects
  
  //Simple Hists
  TH1F ptMC5cal, etaMC5cal, phiMC5cal, m2jMC5cal;
  TH1F ptMC5gen, etaMC5gen, phiMC5gen, m2jMC5gen;
  TH1F ptIC5cal, etaIC5cal, phiIC5cal, m2jIC5cal;
  TH1F ptIC5gen, etaIC5gen, phiIC5gen, m2jIC5gen;
  TH1F ptKT10cal, etaKT10cal, phiKT10cal, m2jKT10cal;
  TH1F ptKT10gen, etaKT10gen, phiKT10gen, m2jKT10gen;
  
  //Calorimeter Sub-System Analysis Histograms for IC5 CaloJets only
  TH1F emEnergyFraction, emEnergyInEB, emEnergyInEE, emEnergyInHF;
  TH1F hadEnergyInHB, hadEnergyInHE, hadEnergyInHF, hadEnergyInHO;
  TProfile EBfractionVsEta, EEfractionVsEta, HBfractionVsEta;
  TProfile HOfractionVsEta, HEfractionVsEta, HFfractionVsEta; 
  TProfile CaloEnergyVsEta, GenEnergyVsEta, emEnergyVsEta, hadEnergyVsEta;
  TProfile CaloErespVsEta, emErespVsEta, hadErespVsEta;
  TProfile WindowEBfractionVsEta, WindowEEfractionVsEta, WindowHBfractionVsEta;
  TProfile WindowHOfractionVsEta, WindowHEfractionVsEta, WindowHFfractionVsEta; 
  TProfile WindowCaloErespVsEta, WindowEmErespVsEta, WindowHadErespVsEta;
  TProfile WindowCaloEnergyVsEta, WindowGenEnergyVsEta, WindowEmEnergyVsEta, WindowHadEnergyVsEta;
  TProfile WindowMaxTowErespVsEta, WindowMaxEmErespVsEta, WindowMaxHadErespVsEta;
  TH2F GenEnergyVsEta2D, AllGenEnergyVsEta2D;

  //Matched jets Analysis Histograms for MC5 CaloJets only
  TH1F dR, dRcor;
  TProfile respVsPt, corRespVsPt;

  //Histo File 
  TFile* m_file;

  //Internal parameters
  int evtCount;
  int numJets;

};

#endif
