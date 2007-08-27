#ifndef DTDigiTask_H
#define DTDigiTask_H

/*
 * \file DTDigiTask.h
 *
 * $Date: 2007/03/22 18:52:01 $
 * $Revision: 1.11 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTSuperLayerId;
class DTLayerId;
class DTChamberId;
class DTTtrig;
class DTT0;


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
  void bookHistos(const DTSuperLayerId& dtSL, std::string folder, std::string histoTag);
  void bookHistos(const DTChamberId& dtCh, std::string folder, std::string histoTag);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// get the L1A source
  std::string triggerSource();

  bool layerExist(DTLayerId lId);

private:

  bool debug;
  int nevents;

  /// no needs to be precise. Value from PSets will always be used
  int tMax;
  int maxTDCHits;

  /// tTrig from the DB
  float tTrig;
  float tTrigRMS;

  //check for sync noise
  //  bool newChamber;
  //  DTChamberId chDone;
  std::map<DTChamberId,int> hitMap;
  std::map<DTChamberId,bool> hitMapCheck;
  int syncNumTot;
  int syncNum;

  edm::Handle<LTCDigiCollection> ltcdigis;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  edm::ESHandle<DTTtrig> tTrigMap;
  edm::ESHandle<DTT0> t0Map;


  std::string outputFile;

  std::map<std::string, std::map<uint32_t, MonitorElement*> > digiHistos;


};

#endif
