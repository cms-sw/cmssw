#ifndef RPCMonitorEfficiency_h
#define RPCMonitorEfficiency_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring using RPCDigi and RPCRecHit.
 *
 *  $Date: 2006/06/06 13:48:26 $
 *  $Revision: 1.2 $
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

class RPCMonitorEfficiency : public edm::EDAnalyzer {
   public:
	explicit RPCMonitorEfficiency( const edm::ParameterSet& );
	~RPCMonitorEfficiency();
   
	virtual void analyze( const edm::Event&, const edm::EventSetup& );

	virtual void endJob(void);
        

   private:
	int counter;
	std::string nameInLog;
	bool saveRootFile;
	int  saveRootFileEventsInterval;
	std::string RootFileName;
	/// back-end interface
	DaqMonitorBEInterface * dbe;
};

#endif
