#ifndef DTDigiTask_H
#define DTDigiTask_H

/*
 * \file DTDigiTask.h
 *
 * $Date: 2006/02/08 21:14:30 $
 * $Revision: 1.3 $
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
  void bookHistos(const DTLayerId& dtLayer, string folder, string histoTag);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);


private:

  int nevents;
  
  int tMaxRescaled, tTrigRescaling;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  string outputFile;
  ofstream logFile;

  map<string, map<int, MonitorElement*> > digiHistos;

  map<int, MonitorElement*> noiseOccupancyHistos;
  map<int, MonitorElement*> inTimeHitsOccupancyHistos;
  map<int, MonitorElement*> afterPulsesOccupancyHistos;
  map<int, MonitorElement*> hitsOrderHistos;
  map<int, MonitorElement*> timeboxHistos;
  map<int, MonitorElement*> cathodPhotoPeakHistos;  


  
};

//DEFINE_FWK_MODULE(DTDigiTask)

#endif
