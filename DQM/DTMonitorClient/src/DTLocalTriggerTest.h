#ifndef DTLocalTriggerTest_H
#define DTLocalTriggerTest_H


/** \class DTLocalTriggerTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/05/18 08:07:47 $
 *  $Revision: 1.1 $
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <vector>
#include <map>

class DTChamberId;

class DTLocalTriggerTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTLocalTriggerTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// book the new ME
  void bookWheelHistos(int wheel, std::string folder);

  /// Get the ME name
  std::string getMEName(std::string histoTag, std::string subfolder, const DTChamberId & chambid);


private:

  int nevents;

  DaqMonitorBEInterface* dbe;
  std::string sourceFolder;
  edm::ParameterSet parameters;
  std::map<int,std::map<std::string,MonitorElement*> > phiME;
  std::map<int,std::map<std::string,MonitorElement*> > thetaME;
  std::map<int,MonitorElement*> segME;

};

#endif
