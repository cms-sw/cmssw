#ifndef EwkDQM_H
#define EwkDQM_H


/** \class EwkDQM
 *
 *  DQM offline for QCD-Photons
 *
 *  $Date: 2009/07/14 15:22:21 $
 *  $Revision: 1.2 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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
  void beginJob(edm::EventSetup const& iSetup);

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

  // Variables from config file
  std::string   theElecTriggerPathToPass;
  std::string   theMuonTriggerPathToPass;
  edm::InputTag theTriggerResultsCollection;
  edm::InputTag theMuonCollectionLabel;
  edm::InputTag theElectronCollectionLabel;
  edm::InputTag theCaloJetCollectionLabel;
  edm::InputTag theCaloMETCollectionLabel;

  // Histograms
  MonitorElement* h_mumu_invMass;
  MonitorElement* h_ee_invMass;
  MonitorElement* h_jet_et;
  MonitorElement* h_jet_count;
//WCP: Adding a Histo
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
  MonitorElement* h_t1_et;
  MonitorElement* h_t1_eta;
  MonitorElement* h_t1_phi;
  MonitorElement* h_met;
  MonitorElement* h_met_phi;
  MonitorElement* h_e_invWMass;
};
#endif
