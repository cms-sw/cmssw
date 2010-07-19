#ifndef BPhysicsOniaDQM_H
#define BPhysicsOniaDQM_H


/** \class BPhysicsOniaDQM
 *
 *  DQM offline for quarkonia
 *
 *  $Date: 2010/01/04 11:50:29 $
 *  $Revision: 1.4 $
 *  \author S. Bolognesi, Eric - CERN
 */

#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class DQMStore;
class MonitorElement;

class BPhysicsOniaDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  BPhysicsOniaDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~BPhysicsOniaDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob();

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
