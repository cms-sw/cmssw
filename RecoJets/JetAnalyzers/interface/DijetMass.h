#ifndef RecoExamples_DijetMass_h
#define RecoExamples_DijetMass_h
#include <TH1.h>
#include <TProfile.h>
#include <TH2.h>
/* \class DijetMass
 *
 * \author Robert Harris
 *
 * \version 1
 *
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"

class DijetMass : public edm::EDAnalyzer {
public:
  DijetMass( const edm::ParameterSet & );

private:
  //Framwework stuff
  void beginJob( const edm::EventSetup & );
  void analyze( const edm::Event& , const edm::EventSetup& );
  void endJob();

  // Parameters passed via the config file
  double PtHistMax;      // Maximum edge of Pt histograms
  std::string GenType;   // Type of Generated Process on Input to analysis
    
  // Root objects
  
  //Simple Hists
  TH1F ptMC5cal, etaMC5cal, phiMC5cal, m2jMC5cal;
  TH1F ptMC5gen, etaMC5gen, phiMC5gen, m2jMC5gen;
  TH1F ptMC5cor, etaMC5cor, phiMC5cor, m2jMC5cor;
  TH1F ptIC5cal, etaIC5cal, phiIC5cal, m2jIC5cal;
  TH1F ptIC5gen, etaIC5gen, phiIC5gen, m2jIC5gen;
  TH1F ptIC5cor, etaIC5cor, phiIC5cor, m2jIC5cor;
  TH1F ptKT10cal, etaKT10cal, phiKT10cal, m2jKT10cal;
  TH1F ptKT10gen, etaKT10gen, phiKT10gen, m2jKT10gen;
  
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
