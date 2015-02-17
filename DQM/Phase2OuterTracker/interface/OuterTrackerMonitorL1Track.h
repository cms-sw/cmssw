#ifndef Phase2OuterTracker_OuterTrackerMonitorL1Track_h
#define Phase2OuterTracker_OuterTrackerMonitorL1Track_h

#include <vector>
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class DQMStore;

class OuterTrackerMonitorL1Track : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorL1Track(const edm::ParameterSet&);
  ~OuterTrackerMonitorL1Track();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  MonitorElement* L1Track_Track_3Stubs_N = 0;
  MonitorElement* L1Track_Track_2Stubs_N = 0;
  
  edm::InputTag tagTTTracks;

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
};
#endif
