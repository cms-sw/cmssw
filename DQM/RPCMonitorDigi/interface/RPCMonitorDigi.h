#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring (strip id, cluster size).
 *
 *  $Date: 2006/09/14 17:09:21 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni (CERN)
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

#include<string>
#include<map>

class RPCDetId;

class RPCMonitorDigi : public edm::EDAnalyzer {
   public:
	explicit RPCMonitorDigi( const edm::ParameterSet& );
	~RPCMonitorDigi();
   
	virtual void analyze( const edm::Event&, const edm::EventSetup& );

	virtual void beginJob(edm::EventSetup const&);
	virtual void endJob(void);
        
	/// Booking of MonitoringElemnt for one RPCDetId (= roll)
	std::map<std::string, MonitorElement*> bookDetUnitME(RPCDetId & detId);
	
	/// Booking of MonitoringElemnt at Wheel/Disk level
	std::map<std::string, MonitorElement*> bookRegionRing(int region, int ring);

   private:
	
	int counter;
	/// back-end interface
	DaqMonitorBEInterface * dbe;
        MonitorElement * GlobalZYHitCoordinates;
        MonitorElement * GlobalZXHitCoordinates;
        MonitorElement * GlobalZPhiHitCoordinates;
        
	std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
        std::map<std::pair<int,int>, std::map<std::string, MonitorElement*> >  meWheelDisk;
	
	std::string nameInLog;
	bool saveRootFile;
	int  saveRootFileEventsInterval;
	std::string RootFileName;
	std::string GlobalHistogramsFolder;
	std::map<uint32_t,bool> foundHitsInChamber;
};

#endif
