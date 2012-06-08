#ifndef EFFICIENCYANALYZER_H
#define EFFICIENCYANALYZER_H

/**   Class EfficiencyAnalyzer
 *  
 *    DQM monitoring for dimuon mass
 *    
 *    Author:  S.Folgueras, A. Calderon
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
  edm::InputTag theTrackCollectionLabel;


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

  MonitorElement* h_passProbes_TightMu_pt;
  MonitorElement* h_passProbes_TightMu_barrel_pt;
  MonitorElement* h_passProbes_TightMu_endcap_pt;
  MonitorElement* h_passProbes_TightMu_eta;
  MonitorElement* h_passProbes_TightMu_hp_eta;
  MonitorElement* h_passProbes_TightMu_phi;

  MonitorElement* h_failProbes_TightMu_pt;
  MonitorElement* h_failProbes_TightMu_eta;
  MonitorElement* h_failProbes_TightMu_phi;

  MonitorElement* h_allProbes_pt;
  MonitorElement* h_allProbes_barrel_pt;
  MonitorElement* h_allProbes_endcap_pt;
  MonitorElement* h_allProbes_eta;
  MonitorElement* h_allProbes_hp_eta;
  MonitorElement* h_allProbes_phi;


  MonitorElement* test_TightMu_Minv;



};
#endif 

