#ifndef EwkDQM_H
#define EwkDQM_H


/** \class EwkDQM
 *
 *  DQM offline for SMP V+Jets
 *
 *  $Date: 2012/06/28 10:29:38 $
 *  $Revision: 1.12 $
 *  \author Valentina Gori, University of Firenze
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"   
#include "FWCore/Framework/interface/EDAnalyzer.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"

class DQMStore;
class MonitorElement;

class EwkDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  EwkDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~EwkDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob();

  ///
  void beginRun( const edm::Run& , const edm::EventSetup& );

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

  double calcDeltaPhi(double phi1, double phi2);

 private:

  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  // Switch for verbosity
  std::string logTraceName;

  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;


  // Variables from config file
  std::vector<std::string>   theElecTriggerPathToPass_;
  std::vector<std::string>   theMuonTriggerPathToPass_;
  //std::vector<std::string> eleTrigPathNames_;
  //std::vector<std::string> muTrigPathNames_;
  edm::InputTag theTriggerResultsCollection_;
  edm::InputTag theMuonCollectionLabel_;
  edm::InputTag theElectronCollectionLabel_;
  //edm::InputTag theCaloJetCollectionLabel;
  edm::InputTag thePFJetCollectionLabel_;
  edm::InputTag theCaloMETCollectionLabel_;

  double eJetMin_;

  // Histograms
  MonitorElement* h_vertex_number;
  MonitorElement* h_vertex_chi2;
  MonitorElement* h_vertex_numTrks;
  MonitorElement* h_vertex_sumTrks;
  MonitorElement* h_vertex_d0;

  MonitorElement* h_jet_count;
  MonitorElement* h_jet_et;
  MonitorElement* h_jet_pt;
  MonitorElement* h_jet_eta;
  MonitorElement* h_jet_phi;

  MonitorElement* h_jet2_et;
  //MonitorElement* h_jet2_pt;
  MonitorElement* h_jet2_eta;
  MonitorElement* h_jet2_phi;

  MonitorElement* h_e1_et;
  MonitorElement* h_e2_et;
  MonitorElement* h_e1_eta;
  MonitorElement* h_e2_eta;
  MonitorElement* h_e1_phi;
  MonitorElement* h_e2_phi;

  MonitorElement* h_m1_pt;
  MonitorElement* h_m2_pt;
  MonitorElement* h_m1_eta;
  MonitorElement* h_m2_eta;
  MonitorElement* h_m1_phi;
  MonitorElement* h_m2_phi;

  //MonitorElement* h_t1_et;
  //MonitorElement* h_t1_eta;
  //MonitorElement* h_t1_phi;

  MonitorElement* h_met;
  MonitorElement* h_met_phi;

  MonitorElement* h_e_invWMass;
  MonitorElement* h_m_invWMass;
  MonitorElement* h_mumu_invMass;
  MonitorElement* h_ee_invMass;
};
#endif
