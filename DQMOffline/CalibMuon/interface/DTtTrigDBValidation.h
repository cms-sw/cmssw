#ifndef DTtTrigDBValidation_H
#define DTtTrigDBValidation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <string>
#include <fstream>
#include <vector>

//class DTTtrig;
class TFile;

class DTtTrigDBValidation : public DQMEDAnalyzer {

public:
  /// Constructor
  DTtTrigDBValidation(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTtTrigDBValidation();

  /// Operations
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze( const edm::Event&, const edm::EventSetup&) override;

private:

  // Switch for verbosity
  std::string metname_;
  // The DB label
  std::string labelDBRef_;
  std::string labelDB_;

  int lowerLimit_;
  int higherLimit_;

  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom_;

  // Map of the tTrig and sigma by super-layer
  std::map<DTSuperLayerId, std::pair<float,float> > tTrigRefMap_;
  std::map<DTSuperLayerId, std::pair<float,float> > tTrigMap_;

  // Map of the tTrig difference histos by (wheel,sector)
  std::map<std::pair<int,int>, MonitorElement* > tTrigDiffHistos_;
  std::map<int, MonitorElement* > tTrigDiffWheel_;

  void bookHistos(DQMStore::IBooker &, int, int);
  void bookHistos(DQMStore::IBooker &, int wheel);
  // Compute the station from the bin number of mean and sigma histos
  int stationFromBin(int bin) const;
  // Compute the sl from the bin number of mean and sigma histos
  int slFromBin(int bin) const;
};
#endif
