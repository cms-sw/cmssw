#ifndef EwkDQM_H
#define EwkDQM_H


/** \class EwkDQM
 *
 *  DQM offline for QCD-Photons
 *
 *  $Date: 2009/06/28 09:46:47 $
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
  std::string   theTriggerPathToPass;
  edm::InputTag theTriggerResultsCollection;
  edm::InputTag theMuonCollectionLabel;
  edm::InputTag theElectronCollectionLabel;
  edm::InputTag theCaloJetCollectionLabel;

  // Histograms
  MonitorElement* h_mumu_invMass;
  MonitorElement* h_ee_invMass;
  MonitorElement* h_jet_et;
  MonitorElement* h_jet_count;
};
#endif
