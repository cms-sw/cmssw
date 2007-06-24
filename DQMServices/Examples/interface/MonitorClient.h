#ifndef MonitorClient_H
#define MonitorClient_H

/** \class MonitorClient
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/06/21 15:05:55 $
 *  $Revision: 1.1.2.1 $
 *  \author  M. Zanetti CERN
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>


class MonitorClient: public edm::EDAnalyzer{

public:

  /// Constructor
  MonitorClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~MonitorClient();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();



private:

  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  MonitorElement * clientHisto;

};

#endif


