#ifndef EFFICIENCYANALYZER_H
#define EFFICIENCYANALYZER_H

/**   Class EfficiencyAnalyzer
 *  
 *    DQM monitoring for dimuon mass
 *    
 *    Author:  S.Folgueras, U. Oviedo
 */

/* Base Class Headers */
#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

class EfficiencyAnalyzer : public MuonAnalyzerBase {
  
 public:
  /* Constructor */ 
  EfficiencyAnalyzer(const edm::ParameterSet& pset, MuonServiceProxy *theService);
  
  /* Destructor */ 
  virtual ~EfficiencyAnalyzer() ;

  /* Operations */ 
  void beginJob (DQMStore *dbe);
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  //  void endJob ();

 protected:
  edm::ParameterSet parameters;
  
  // Switch for verbosity
  std::string metname;
  
  // STA Label
  edm::InputTag theSTACollectionLabel;
  edm::InputTag theMuonCollectionLabel;

  //histo binning parameters
  int etaBin_;
  int phiBin_;
  int ptBin_;
  double ptMin_;  
  double ptMax_;
  double etaMin_;  
  double etaMax_;
  double phiMin_;  
  double phiMax_;

  MonitorElement* Eff_Numerator_pt;
  MonitorElement* Eff_Numerator_eta;
  MonitorElement* Eff_Numerator_phi;

  MonitorElement* Eff_Denominator_pt;
  MonitorElement* Eff_Denominator_eta;
  MonitorElement* Eff_Denominator_phi;
};
#endif 

