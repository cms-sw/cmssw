#ifndef QcdPhotonsDQM_H
#define QcdPhotonsDQM_H


/** \class QcdPhotonsDQM
 *
 *  DQM offline for QCD-Photons
 *
 *  $Date: 2012/10/10 04:00:00 $
 *  $Revision: 1.18 $
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

  // Divide histograms
  void endRun(const edm::Run&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

 private:

  // ----------member data ---------------------------
  
  DQMStore* theDbe;

  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;

  // Switch for verbosity
  std::string logTraceName;

  // Variables from config file
  std::string   theTriggerPathToPass_;
  std::vector<std::string> thePlotTheseTriggersToo_;
  edm::InputTag trigTag_;
  edm::InputTag thePhotonCollectionLabel_;
  edm::InputTag theJetCollectionLabel_;
  edm::InputTag theVertexCollectionLabel_;
  double theMinJetPt_;
  double theMinPhotonEt_;
  bool   theRequirePhotonFound_;
  double thePlotPhotonMaxEt_;
  double thePlotPhotonMaxEta_;
  double thePlotJetMaxEta_;
 
  edm::InputTag theBarrelRecHitTag;
  edm::InputTag theEndcapRecHitTag;

  int num_events_in_run;

  // Histograms
  MonitorElement* h_triggers_passed;
  MonitorElement* h_photon_et_beforeCuts;
  MonitorElement* h_photon_et;
  MonitorElement* h_photon_eta;
  MonitorElement* h_photon_count_bar;
  MonitorElement* h_photon_count_end;
  MonitorElement* h_jet_pt;
  MonitorElement* h_jet_eta;
  MonitorElement* h_jet_count;
  MonitorElement* h_deltaPhi_photon_jet;
  MonitorElement* h_deltaPhi_jet_jet2;
  MonitorElement* h_deltaEt_photon_jet;
  MonitorElement* h_jet2_ptOverPhotonEt;
  MonitorElement* h_jet2_pt;
  MonitorElement* h_jet2_eta;
  MonitorElement* h_deltaPhi_photon_jet2;
  MonitorElement* h_deltaR_jet_jet2;
  MonitorElement* h_deltaR_photon_jet2;

  MonitorElement* h_photon_et_jetcs;
  MonitorElement* h_photon_et_jetco;
  MonitorElement* h_photon_et_jetfs;
  MonitorElement* h_photon_et_jetfo;

  MonitorElement* h_photon_et_ratio_co_cs;
  MonitorElement* h_photon_et_ratio_fo_fs;
  MonitorElement* h_photon_et_ratio_cs_fs;
  MonitorElement* h_photon_et_ratio_co_fs;
  MonitorElement* h_photon_et_ratio_cs_fo;
  MonitorElement* h_photon_et_ratio_co_fo;
};
#endif
