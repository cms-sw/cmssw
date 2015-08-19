#ifndef DQMDbHarvester_H
#define DQMDbHarvester_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// // Trigger
// #include "DataFormats/Common/interface/TriggerResults.h"
// #include "DataFormats/HLTReco/interface/TriggerObject.h"
// #include "DataFormats/HLTReco/interface/TriggerEvent.h"
// #include "FWCore/Common/interface/TriggerNames.h"
 
//CORAL includes
#include "RelationalAccess/ConnectionService.h"

//STL includes
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <initializer_list>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <ctime>
#include <cmath>

//histogram's values per run
struct valuesOfHistogram {
  int test_run;
  int test_entries;
  double test_x_mean;
  double test_x_mean_error;
  double test_x_rms;
  double test_x_rms_error;

  double test_y_mean;
  double test_y_mean_error;
  double test_y_rms;
  double test_y_rms_error;

  double test_z_mean;
  double test_z_mean_error;
  double test_z_rms;
  double test_z_rms_error;
};

class DQMDbHarvester: public DQMEDHarvester{

public:

  DQMDbHarvester(const edm::ParameterSet& ps);
  virtual ~DQMDbHarvester();
  
protected:

  void beginJob();
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&);  //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  //Parse histograms that should be treated as run based
  //It is neccessary to gather data from every lumi, so it cannot be done in the endRun
  void dqmDbRunInitialize(std::vector < std::pair <MonitorElement *, valuesOfHistogram> > & histograms);

  //Gather data from lumisections
  void dqmDbRunProcess(int run);

  //Drop all the data from a run into the database
  void dqmDbRunDrop();

  void dqmDbLumiDrop(std::vector <MonitorElement *> & histograms, int luminosity, int run);
  coral::ConnectionService m_connectionService;

  std::unique_ptr<coral::ISessionProxy> m_session;
  std::string m_connectionString;

private:
  //variables from config file
  std::string numMonitorName_;
  std::string denMonitorName_;

  // Histograms
  MonitorElement* h_ptRatio;

//  std::vector <MonitorElement *> histogramsPerLumi;
  std::vector < std::pair <MonitorElement *, valuesOfHistogram> > histogramsPerRun;
};


#endif
