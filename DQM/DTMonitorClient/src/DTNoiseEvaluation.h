#ifndef DTNoiseEvaluation_H
#define DTNoiseEvaluation_H

/*
 * \file DTNoiseEvaluation.h
 *
 * $Date: 2006/08/10 16:27:18 $
 * $Revision: 1.2 $
 * \author M. Zanetti - INFN Padova
 *
*/

//#include "PluginManager/ModuleDef.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>



#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/Core/interface/DQMDefinitions.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTLayerId;
class DTWireId;



class DTNoiseEvaluation: public edm::EDAnalyzer{

public:
  
  /// Constructor
  DTNoiseEvaluation(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTNoiseEvaluation();

protected:

  /// BeginJob (needed?)
  void beginJob(const edm::EventSetup& c);

  /// Book the ME
  void bookHistos(const DTLayerId& dtLayer, std::string histoTag);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// End job. Here write the bad channels on the DB
  void endJob();

  /// create the quality tests
  void createQualityTests(void);

  /// tune cuts for quality tests
  void tuneCuts(void);

  /// run quality tests;
  void runDQMTest(void);
  void runStandardTest(void);

  /// show channels that failed test
  void drawSummaryNoise();

private:

  int nevents;
  std::string outputFile;
  std::string criterionName;

  edm::ParameterSet parameters;

  edm::ESHandle<DTGeometry> muonGeom;

  // back-end interface
  DaqMonitorBEInterface * dbe;

  // Monitor UI
  MonitorUserInterface * mui;

  // histograms: < DetID, Histogram >
  std::map< uint32_t , MonitorElement* > occupancyHistos;

  // collection of histograms' names
  std::vector<std::string> histoNamesCollection;

  // the collection of noisy channels
  std::vector<DTWireId> theNoisyChannels;
  
  std::map<DTLayerId, float> noiseStatistics;

  std::map< uint32_t , MonitorElement* > noiseAverageHistos;

  // quality tests
  NoisyChannelROOT* theNoiseTest;

};


#endif
