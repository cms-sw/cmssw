#ifndef MuonEnergyDepositAnalyzer_H
#define MuonEnergyDepositAnalyzer_H


/** \class MuEnergyDepositAnalyzer
 *
 *  DQM monitoring source for muon energy deposits
 *
 *  $Date: 2008/03/13 11:10:41 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"



class MuonEnergyDepositAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor
  MuonEnergyDepositAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonEnergyDepositAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DaqMonitorBEInterface * dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon& recoMu);


 private:
  // ----------member data ---------------------------
  
  DaqMonitorBEInterface * dbe;
  edm::ParameterSet parameters;
  // Switch for verbosity
  bool debug;
  std::string metname;
  // STA Label
  edm::InputTag theSTACollectionLabel;

  //histo binning parameters
  int emNoBin;
  double emNoMin;
  double emNoMax;

  int emS9NoBin;
  double emS9NoMin;
  double emS9NoMax;

  int hadNoBin;
  double hadNoMin;
  double hadNoMax;

  int hadS9NoBin;
  double hadS9NoMin;
  double hadS9NoMax;

  int hoNoBin;
  double hoNoMin;
  double hoNoMax;

  int hoS9NoBin;
  double hoS9NoMin;
  double hoS9NoMax;

  //the histos
  MonitorElement * ecalDepEnergy;
  MonitorElement * ecalS9DepEnergy;
  MonitorElement * hcalDepEnergy;
  MonitorElement * hcalS9DepEnergy;
  MonitorElement * hoDepEnergy;
  MonitorElement * hoS9DepEnergy;

};
#endif  
