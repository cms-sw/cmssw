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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EfficiencyAnalyzer : public DQMEDAnalyzer {
  
 public:
  /* Constructor */ 
  EfficiencyAnalyzer(const edm::ParameterSet& pset);
  
  /* Destructor */ 
  virtual ~EfficiencyAnalyzer() ;

  /* Operations */ 
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  edm::ParameterSet parameters;
  MuonServiceProxy *theService;
    
  // Switch for verbosity
  std::string metname;
  
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

  int vtxBin_;
  double vtxMin_;
  double vtxMax_;

  std::string ID_;

  MonitorElement* h_passProbes_ID_pt;
  MonitorElement* h_passProbes_ID_EB_pt;
  MonitorElement* h_passProbes_ID_EE_pt;
  MonitorElement* h_passProbes_ID_eta;
  MonitorElement* h_passProbes_ID_hp_eta;
  MonitorElement* h_passProbes_ID_phi;
  MonitorElement* h_passProbes_detIsoID_pt;
  MonitorElement* h_passProbes_EB_detIsoID_pt;
  MonitorElement* h_passProbes_EE_detIsoID_pt;
  MonitorElement* h_passProbes_pfIsoID_pt;
  MonitorElement* h_passProbes_EB_pfIsoID_pt;
  MonitorElement* h_passProbes_EE_pfIsoID_pt;
  MonitorElement* h_passProbes_detIsoID_nVtx; 
  MonitorElement* h_passProbes_pfIsoID_nVtx; 
  MonitorElement* h_passProbes_EB_detIsoID_nVtx; 
  MonitorElement* h_passProbes_EE_detIsoID_nVtx; 
  MonitorElement* h_passProbes_EB_pfIsoID_nVtx; 
  MonitorElement* h_passProbes_EE_pfIsoID_nVtx; 

  MonitorElement* h_failProbes_ID_pt;
  MonitorElement* h_failProbes_ID_eta;
  MonitorElement* h_failProbes_ID_phi;

  MonitorElement* h_allProbes_pt;
  MonitorElement* h_allProbes_EB_pt;
  MonitorElement* h_allProbes_EE_pt;
  MonitorElement* h_allProbes_eta;
  MonitorElement* h_allProbes_hp_eta;
  MonitorElement* h_allProbes_phi;
  MonitorElement* h_allProbes_ID_pt;
  MonitorElement* h_allProbes_EB_ID_pt;
  MonitorElement* h_allProbes_EE_ID_pt;
  MonitorElement* h_allProbes_ID_nVtx;
  MonitorElement* h_allProbes_EB_ID_nVtx;
  MonitorElement* h_allProbes_EE_ID_nVtx;
  
  
  // Apply deltaBeta PU corrections to the PF isolation eficiencies.
  MonitorElement* h_passProbes_pfIsodBID_pt;
  MonitorElement* h_passProbes_EB_pfIsodBID_pt;
  MonitorElement* h_passProbes_EE_pfIsodBID_pt;
  MonitorElement* h_passProbes_pfIsodBID_nVtx;
  MonitorElement* h_passProbes_EB_pfIsodBID_nVtx; 
  MonitorElement* h_passProbes_EE_pfIsodBID_nVtx; 

  int _numPV;

  // STA Label
  edm::EDGetTokenT<reco::MuonCollection>  theMuonCollectionLabel_;
  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionLabel_;

  //Vertex requirements
  bool doPVCheck_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot>         theBeamSpotLabel_;
};
#endif 

