#ifndef DTtTrigDBValidation_H
#define DTtTrigDBValidation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <string>
#include <fstream>
#include <vector>

//class DTTtrig;
class TFile;

class DTtTrigDBValidation : public edm::EDAnalyzer {

public:
  /// Constructor
  DTtTrigDBValidation(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTtTrigDBValidation();

  /// Operations
  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);
  void analyze(edm::Event const&, edm::EventSetup const&) {}
  void endJob();

private:

  // Switch for verbosity
  std::string metname;
  // The DB label
  std::string labelDBRef;
  std::string labelDB;

  int lowerLimit;
  int higherLimit;

  // The file which will contain the difference plots
  bool outputMEsInRootFile;
  std::string outputFileName;

  DQMStore* dbe;
  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom;

  // Map of the tTrig and sigma by super-layer
  std::map<DTSuperLayerId, std::pair<float,float> > tTrigRefMap;
  std::map<DTSuperLayerId, std::pair<float,float> > tTrigMap;

  // Map of the tTrig difference histos by (wheel,sector)
  std::map<std::pair<int,int>, MonitorElement* > tTrigDiffHistos;
  std::map<int, MonitorElement* > tTrigDiffWheel;

  void bookHistos(int,int);
  void bookHistos(int wheel);
  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;
};
#endif
