#ifndef DTChamberEfficiencyTest_H
#define DTChamberEfficiencyTest_H


/** \class DTChamberEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/04/24 09:35:49 $
 *  $Revision: 1.1 $
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

class DTChamberEfficiencyTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTChamberEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTChamberEfficiencyTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookHistos(const DTChamberId & ch);

  /// Get the ME name
  std::string getMEName(std::string histoTag, const DTChamberId & chID);


private:

  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  std::map< std::string , MonitorElement* > xEfficiencyHistos;
  std::map< std::string , MonitorElement* > yEfficiencyHistos;
  std::map< std::string , MonitorElement* > xVSyEffHistos;

};

#endif
