#ifndef DTEfficiencyTest_H
#define DTEfficiencyTest_H


/** \class DTEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/03/30 16:10:06 $
 *  $Revision: 1.2 $
 *  \author  G. Mila - INFN Torino
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
class DTLayerId;

class DTEfficiencyTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTEfficiencyTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(const DTLayerId & ch, int firstWire, int lastWire);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTLayerId & lID);


private:

  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  std::map< std::string , MonitorElement* > EfficiencyHistos;
  std::map< std::string , MonitorElement* > UnassEfficiencyHistos;
  
};

#endif
