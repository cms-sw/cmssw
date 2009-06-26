#ifndef BPhysicsOniaDQM_H
#define BPhysicsOniaDQM_H


/** \class BPhysicsOniaDQM
 *
 *  DQM offline for quarkonia
 *
 *  $Date: 2009/06/18 15:32:53 $
 *  $Revision: 1.1 $
 *  \author S. Bolognesi, Eric - CERN
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class DQMStore;
class MuonServiceProxy;
class MonitorElement;

class BPhysicsOniaDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  BPhysicsOniaDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~BPhysicsOniaDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

 private:

  float computeMass(const math::XYZVector &vec1,const math::XYZVector &vec2);

  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  // Switch for verbosity
  std::string metname;

  // Muon Label
  edm::InputTag theMuonCollectionLabel;

  //The histos
  MonitorElement* diMuonMass_global;
  MonitorElement* diMuonMass_tracker;
  MonitorElement* diMuonMass_standalone;
  MonitorElement* global_background;
  MonitorElement* tracker_background;
  MonitorElement* standalone_background;
};
#endif  
