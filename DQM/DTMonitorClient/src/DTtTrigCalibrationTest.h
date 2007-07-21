#ifndef DTtTrigCalibrationTest_H
#define DTtTrigCalibrationTest_H

/** \class DTtTrigCalibrationTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/05/22 07:15:56 $
 *  $Revision: 1.2 $
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

class DTGeometry;
class DTChamberId;
class DTSuperLayerId;
class DTTtrig;
class DTTimeBoxFitter;

class DTtTrigCalibrationTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTtTrigCalibrationTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTtTrigCalibrationTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(const DTChamberId & ch);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);


private:

  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  DTTimeBoxFitter *theFitter;

  // histograms: < detRawID, Histogram >
  std::map<  uint32_t , MonitorElement* > histos;

};

#endif
