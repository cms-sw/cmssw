#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring (strip id, cluster size).
 *
 *  $Date: 2006/02/02 16:05:13 $
 *  $Revision: 1.3 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include<string>
#include<map>

class RPCDetId;

class RPCMonitorDigi : public edm::EDAnalyzer {
   public:
	explicit RPCMonitorDigi( const edm::ParameterSet& );
	~RPCMonitorDigi();
   
	virtual void analyze( const edm::Event&, const edm::EventSetup& );

	virtual void endJob(void);
        
	/// Booking of MonitoringElemnt for one RPCDetId (= roll)
	std::map<std::string, MonitorElement*> bookDetUnitME(RPCDetId & detId);

   private:
	
	int counter;
	/// back-end interface
	DaqMonitorBEInterface * dbe;
        MonitorElement * h1;
        std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
	
	std::string nameInLog;
};

#endif
