#ifndef MuonEnergyDepositAnalyzer_H
#define MuonEnergyDepositAnalyzer_H


/** \class MuEnergyDepositAnalyzer
 *
 *  DQM monitoring source for muon energy deposits
 *
 *  $Date: 2008/03/27 10:01:58 $
 *  $Revision: 1.4 $
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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"


class MuonEnergyDepositAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor (histo booking)
  MuonEnergyDepositAnalyzer(const edm::ParameterSet&, MuonServiceProxy *theService, DQMStore *theDbe);
  
  /// Destructor
  virtual ~MuonEnergyDepositAnalyzer();
  
  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon& recoMu);


 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
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
