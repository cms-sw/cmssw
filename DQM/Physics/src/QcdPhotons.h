#ifndef QcdPhotons_H
#define QcdPhotons_H


/** \class QcdPhotons
 *
 *  DQM offline for QCD-Photons
 *
 *  $Date: 2009/06/24 14:28:40 $
 *  $Revision: 1.1 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

class DQMStore;
class MonitorElement;

class QcdPhotons : public edm::EDAnalyzer {
 public:

  /// Constructor
  QcdPhotons(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~QcdPhotons();
  
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

  // Collections Labels from config file
  edm::InputTag thePhotonCollectionLabel;
  edm::InputTag theCaloJetCollectionLabel;

  // Histograms
  MonitorElement* h_photon_et;
  MonitorElement* h_jet_et;
  MonitorElement* h_deltaPhi_photon_jet;
  MonitorElement* h_deltaEt_photon_jet;
};
#endif  
