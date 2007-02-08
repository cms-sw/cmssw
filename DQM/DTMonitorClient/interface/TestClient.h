#ifndef TestClient_H
#define TestClient_H

/** \class TestClient
 * *
 *  DQM Test Client
 *
 *  $Date: 2006/06/30 15:31:05 $
 *  $Revision: 1.4 $
 *  \author S. Bolognesi - M. Zanetti 
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
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

class DTGeometry;
class DTTtrig;

class TestClient: public edm::EDAnalyzer{

public:

  /// Constructor
  TestClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~TestClient();

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

  bool debug;
  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  // histograms: < detRawID, Histogram >
  std::map<  uint32_t , MonitorElement* > histos;

};

#endif
