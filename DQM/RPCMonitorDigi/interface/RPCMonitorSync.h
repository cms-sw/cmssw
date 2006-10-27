#ifndef RPCMonitorSync_h
#define RPCMonitorSync_h

/** \class RPCMonitorSync
 *
 * RPC Synchronization Monitoring Class
 *
 *  $Date: 2006/10/27 07:57:31 $
 *  $Revision: 0.1 $
 *
 * \author Piotr Traczyk (SINS)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>
#include <map>

struct timing{

  int early;
  int inTime;
  int late;

  float earlyFraction() const{ 
    return (float)early/(early+inTime+late);  
  }

  float inTimeFraction() const { 
    return (float)inTime/(early+inTime+late);  
  }

  float lateFraction() const { 
    return (float)late/(early+inTime+late);  
  }

  float outOfTimeFraction() const { 
    return (float)(early+late)/(early+inTime+late);  
  }
  
  float offset() const {
    return (float)(late-early)/(early+inTime+late);  
  }

  float width() const {
    return (float)(late+early)/(early+inTime+late);  
  }

};

class RPCDetId;

class RPCMonitorSync : public edm::EDAnalyzer {
  public:
    explicit RPCMonitorSync( const edm::ParameterSet& );
	    ~RPCMonitorSync();
   
    virtual void analyze( const edm::Event&, const edm::EventSetup& );

    virtual void endJob(void);
        
/// Booking of MonitoringElemnt for one RPCDetId (= roll)
    std::map<std::string, MonitorElement*> bookDetUnitME(RPCDetId & detId);

  private:
	
    void readRPCDAQStrips(const edm::Event& iEvent);

    MonitorElement *barrelOffsetHist( char *name, char *title );
    MonitorElement *endcapOffsetHist( char *name, char *title );
    MonitorElement *barrelWidthHist( char *name, char *title );
    MonitorElement *endcapWidthHist( char *name, char *title );

    std::map<int,timing> synchroMap;
    int counter;
	/// back-end interface
    DaqMonitorBEInterface * dbe;
    MonitorElement * h1;
    std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
		
    std::string nameInLog;
    bool saveRootFile;
    int  saveRootFileEventsInterval;
    std::string RootFileName;
};

#endif
