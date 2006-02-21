#ifndef DTDigiClient_H
#define DTDigiClient_H

/*
 * \file DTDigiClient.h
 *
 * $Date: 2006/01/13 12:18:46 $
 * $Revision: 1.32 $
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

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"


#include "TROOT.h"
#include "TGaxis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class DTTtrig;
class DTChamberId;

using namespace cms;
using namespace std;

class DTDigiClient: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDigiClient(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTDigiClient();

protected:

  /// Begin Job 
  void beginJob(const edm::EventSetup& context);

  /// Receive the ME and eventually move to next cycle
  bool receiveMonitoring();

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& context);

  /// Get the noisy channels and the average noise level
  void noiseAnalysis(DTChamberId dtChId, const edm::EventSetup& context);
  
  /// Compare the Noise with the DB
  void checkNoise();

  /// Get the dead channles and the averege number of counts
  void inTimeHitsAnalysis(DTChamberId dtChId);

  /// Check for disconnected cathods and get the average after pulse rate
  void afterPulsesAnalysis(DTChamberId dtChId);

  /// Calibrate the tTrig. Check for disconnected cathods (maybe set another class for this)
  void timeBoxAnalysis();

  /// Set ROOT Style
  void setROOTStyle();



private:

  edm::ParameterSet parameters;

  edm::ESHandle<DTTtrig> tTrigMap;

  int updates;
  int last_operation;

  int numOfEvents;

  string outputFile;

  /// < type_of_problem, <channel, content> >
  map<string, map<int, int> > badChannels; 

  MonitorUserInterface* dtMUI;


};

#endif
