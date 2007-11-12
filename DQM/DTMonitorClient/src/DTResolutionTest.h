#ifndef DTResolutionTest_H
#define DTResolutionTest_H


/** \class DTResolutionTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/04/12 07:34:15 $
 *  $Revision: 1.4 $
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

class DTResolutionTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTResolutionTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTResolutionTest();

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
  std::string getMEName(const DTSuperLayerId & slID);


private:

  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  // histograms: < detRawID, Histogram >
  std::map< std::string , MonitorElement* > MeanHistos;
  std::map< std::string , MonitorElement* > SigmaHistos;
  std::map< std::string , MonitorElement* > MeanHistosSetRange;
  std::map< std::string , MonitorElement* > SigmaHistosSetRange;
  std::map< std::string , MonitorElement* > MeanHistosSetRange2D;
  std::map< std::string , MonitorElement* > SigmaHistosSetRange2D;

};

#endif
