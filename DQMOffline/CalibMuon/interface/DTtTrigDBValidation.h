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
  //Read the DTGeometry and the t0 DB
  void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  void analyze(const edm::Event& event, const edm::EventSetup& setup) {}
  //Do the real work
  void endJob();
  void bookHistos(int,int);
  void bookHistos(int wheel);

protected:

private:

  DQMStore* dbe;
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // The DB label
  std::string labelDBRef;
  std::string labelDB;
  // The file which will contain the difference plot
  std::string outputFileName;

  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom;

  // Map of the tTrig and sigma by super-layer
  std::map<DTSuperLayerId, std::pair<float,float> > tTrigRefMap;
  std::map<DTSuperLayerId, std::pair<float,float> > tTrigMap;

  // Map of the tTrig difference histos by (wheel,sector)
  std::map<std::pair<int,int>, MonitorElement* > tTrigDiffHistos;

  // summary histos
  std::map<int, MonitorElement* > wheelSummary;

  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;

};
#endif
