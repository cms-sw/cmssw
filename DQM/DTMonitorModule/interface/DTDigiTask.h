#ifndef DTDigiTask_H
#define DTDigiTask_H

/*
 * \file DTDigiTask.h
 *
 * $Date: 2006/09/20 14:37:40 $
 * $Revision: 1.6 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

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
class DTLayerId;
class DTSuperLayerId;
class DTChamberId;
class DTTtrig;
class DTT0;


using namespace edm;
using namespace cms;
using namespace std;


class DTDigiTask: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDigiTask(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTDigiTask();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Book the ME
  void bookHistos(const DTLayerId& dtLayer, string histoTag);
  void bookHistos(const DTSuperLayerId& dtSL, string folder, string histoTag);
  void bookHistos(const DTChamberId& dtCh, string folder, string histoTag);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// get the L1A source
  string triggerSource();

private:

  int nevents;

  /// no needs to be precise. Value from PSets will always be used
  int tMax;

  /// tTrig from the DB
  float tTrig;
  float tTrigRMS;

  edm::Handle<LTCDigiCollection> ltcdigis;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTTtrig> tTrigMap;
  edm::ESHandle<DTT0> t0Map;


  string outputFile;
  ofstream logFile;

  map<string, map<uint32_t, MonitorElement*> > digiHistos;

  
};

#endif
