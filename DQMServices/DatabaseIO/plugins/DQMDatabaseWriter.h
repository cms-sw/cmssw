#ifndef DQMSERVICES_DATABASEIO_PLUGINS_DQMDATABASEWRITER_H
#define DQMSERVICES_DATABASEIO_PLUGINS_DQMDATABASEWRITER_H

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

struct HistogramValues {
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



class DQMDatabaseWriter {

public:

  DQMDatabaseWriter(const edm::ParameterSet& ps);
  virtual ~DQMDatabaseWriter();
  
  void initDatabase();

  //Parse histograms that should be treated as run based
  //It is neccessary to gather data from every lumi, so it cannot be done in the endRun
  void dqmDbRunInitialize(std::vector < std::pair <MonitorElement *, HistogramValues> > & histograms);

  //Drop all the data from a run into the database
  void dqmDbRunDrop();

  void dqmDbLumiDrop(std::vector <MonitorElement *> & histograms, int luminosity, int run);

private:
  // Gather data from lumisections for the run statistics
  void processLumi(int run);

protected:
  coral::ConnectionService m_connectionService;
  std::unique_ptr<coral::ISessionProxy> m_session;
  std::string m_connectionString;

  std::vector < std::pair <MonitorElement *, HistogramValues> > histogramsPerRun;
};


#endif
