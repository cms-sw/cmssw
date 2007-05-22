#ifndef DTNoiseTest_H
#define DTNoiseTest_H

/** \class DTNoiseTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/05/15 17:21:35 $
 *  $Revision: 1.1 $
 *  \author  M. Zanetti CERN
 *
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include <CondFormats/DataRecord/interface/DTStatusFlagRcd.h>
#include <CondFormats/DTObjects/interface/DTStatusFlag.h>


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTSuperLayerId;
class DTLayerId ;
class DTWireId;

class DTNoiseTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTNoiseTest(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTNoiseTest();

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
  void bookHistos(const DTChamberId & ch, std::string folder, std::string histoTag);

  /// Get the ME name
  std::string getMEName(const DTSuperLayerId & slID);
  std::string getMEName(const DTChamberId & ch);

private:

  int nevents;
  int updates;
  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ESHandle<DTTtrig> tTrigMap;

  // the collection of noisy channels
  //std::map< uint32_t, std::vector<DTWireId> > theNoisyChannels;
   
  std::vector<DTWireId>  theNoisyChannels;

  // histograms: < detRawID, Histogram >
  //std::map<  uint32_t , MonitorElement* > histos;
  std::map<std::string, std::map<uint32_t, MonitorElement*> > histos;

};

#endif
