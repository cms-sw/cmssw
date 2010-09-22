#ifndef QcdPhotonsDQM_H
#define QcdPhotonsDQM_H


/** \class QcdPhotonsDQM
 *
 *  DQM offline for QCD-Photons
 *
 *  $Date: 2010/01/04 14:46:10 $
 *  $Revision: 1.9 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

class DQMStore;
class MonitorElement;

class QcdPhotonsDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  QcdPhotonsDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~QcdPhotonsDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob();

  ///
  void beginRun( const edm::Run& , const edm::EventSetup& );

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

  float calcDeltaR(float eta1, float phi1, float eta2, float phi2);
  float calcDeltaPhi(float phi1, float phi2);

 private:

  // ----------member data ---------------------------
  
  DQMStore* theDbe;

  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;

  // Switch for verbosity
  std::string logTraceName;

  // Variables from config file
  std::string   theTriggerPathToPass;
  std::vector<std::string> thePlotTheseTriggersToo;
  std::string   theHltMenu;
  std::string   theTriggerResultsCollection;
  edm::InputTag thePhotonCollectionLabel;
  edm::InputTag theCaloJetCollectionLabel;
  int    theMinCaloJetEt;
  int    theMinPhotonEt;
  bool   theRequirePhotonFound;
  double thePlotMaxEt;
  double thePlotPhotonMaxEta;
  double thePlotJetMaxEta;

  // Histograms
  MonitorElement* h_triggers_passed;
  MonitorElement* h_photon_et;
  MonitorElement* h_photon_eta;
  MonitorElement* h_photon_phiMod;
  MonitorElement* h_photon_count;
  MonitorElement* h_jet_et;
  MonitorElement* h_jet_eta;
  MonitorElement* h_jet_count;
  MonitorElement* h_deltaPhi_photon_jet;
  MonitorElement* h_deltaPhi_jet_jet2;
  MonitorElement* h_deltaEt_photon_jet;
  MonitorElement* h_jet2_etOverPhotonEt;
  MonitorElement* h_jet2_et;
  MonitorElement* h_jet2_eta;
  MonitorElement* h_deltaPhi_photon_jet2;
  MonitorElement* h_deltaR_jet_jet2;
  MonitorElement* h_deltaR_photon_jet2;
};
#endif
