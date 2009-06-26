#ifndef QcdPhotonsPhotonJetDQM_H
#define QcdPhotonsPhotonJetDQM_H


/** \class QcdPhotonsPhotonJetDQM
 *
 *  DQM offline for QCD-Photons
 *
 *  $Date: 2009/06/26 13:54:49 $
 *  $Revision: 1.3 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

class DQMStore;
class MonitorElement;

class QcdPhotonsPhotonJetDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  QcdPhotonsPhotonJetDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~QcdPhotonsPhotonJetDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

  double calcDeltaPhi(double phi1, double phi2);

 private:

  float computeMass(const math::XYZVector &vec1,const math::XYZVector &vec2);

  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  // Switch for verbosity
  std::string logTraceName;

  // Variables from config file
  std::string   theTriggerPathToPass;
  edm::InputTag theTriggerResultsCollection;
  edm::InputTag thePhotonCollectionLabel;
  edm::InputTag theCaloJetCollectionLabel;

  // Histograms
  MonitorElement* h_photon_et;
  MonitorElement* h_jet_et;
  MonitorElement* h_jet_count;
  MonitorElement* h_deltaPhi_photon_jet;
  MonitorElement* h_deltaEt_photon_jet;
  MonitorElement* h_jet2_etOverPhotonEt;
};
#endif
