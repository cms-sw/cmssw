#ifndef RPCMonitorSync_h
#define RPCMonitorSync_h

/** \class RPCMonitorSync
 *
 * RPC Synchronization Monitoring Class
 *
 *  $Date: 2009/08/27 09:42:37 $
 *  $Revision: 1.6 $
 *
 * \author Piotr Traczyk (SINS)
 * 
 * \modified Anna Cimmino (INFN)
 *          
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"

#include <string>
#include <map>
#include <math.h>

struct timing{

  int early_all[4];
  int inTime;
  int late_all[4];

  int early() const {
    return (early_all[0]+early_all[1]+early_all[2]+early_all[3]);
  }

  int late() const {
    return (late_all[0]+late_all[1]+late_all[2]+late_all[3]);
  }

  int early_w() const {
    return (early_all[0]+2*early_all[1]+3*early_all[2]+4*early_all[3]);
  }

  int late_w() const {
    return (late_all[0]+2*late_all[1]+3*late_all[2]+4*late_all[3]);
  }

  int early_w2() const {
    return (early_all[0]+4*early_all[1]+9*early_all[2]+16*early_all[3]);
  }

  int late_w2() const {
    return (late_all[0]+4*late_all[1]+9*late_all[2]+16*late_all[3]);
  }

  float earlyFraction() const{ 
    return (float)early()/(early()+inTime+late());  
  }

  float inTimeFraction() const { 
    return (float)inTime/(early()+inTime+late());  
  }

  float lateFraction() const { 
    return (float)late()/(early()+inTime+late());  
  }

  float outOfTimeFraction() const { 
    return (float)(early()+late())/(early()+inTime+late());  
  }
  
  float offset() const {
    return (float)(late_w()-early_w())/(early()+inTime+late());  
  }

  float width() const {
    return (float)sqrt((float)(late_w2()+early_w2())/(early()+inTime+late()));  
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

    MonitorElement *barrelOffsetHist( std::string name, std::string title );
    MonitorElement *endcapOffsetHist( std::string name, std::string title );
    MonitorElement *barrelWidthHist( std::string name, std::string title );
    MonitorElement *endcapWidthHist( std::string name, std::string title );

    std::map<uint32_t,timing> synchroMap;
    int counter;
	/// back-end interface
    DQMStore * dbe;
    MonitorElement * h1;
    std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
		
    std::string nameInLog;
    bool saveRootFile;
    int  saveRootFileEventsInterval;
    std::string RootFileName;
};

#endif
