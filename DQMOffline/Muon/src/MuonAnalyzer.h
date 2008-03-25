#ifndef MuonAnalyzer_H
#define MuonAnalyzer_H


/** \class MuonAnalyzer
 *
 *  DQM muon analysis monitoring
 *
 *  $Date: 2008/03/18 12:01:35 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMOffline/Muon/src/MuonEnergyDepositAnalyzer.h"
#include "DQMOffline/Muon/src/MuonSeedsAnalyzer.h"


class MuonAnalyzer : public edm::EDAnalyzer {
 public:

  /// Constructor
  MuonAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

 private:
  // ----------member data ---------------------------
  
  DaqMonitorBEInterface * dbe;
  edm::ParameterSet parameters;

  // STA Label
  edm::InputTag theSTACollectionLabel;
  // Seed Label
  edm::InputTag theSeedsCollectionLabel;
  
  // the muon energy analyzer
  MuonEnergyDepositAnalyzer * theMuEnergyAnalyzer;
  // the seeds analyzer
  MuonSeedsAnalyzer * theSeedsAnalyzer;

};
#endif  
