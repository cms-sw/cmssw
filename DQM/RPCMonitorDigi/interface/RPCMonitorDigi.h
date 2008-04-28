#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring (strip id, cluster size).
 *
 *  $Date: 2008/04/25 19:34:15 $
 *  $Revision: 1.13 $
 *
 * \author Ilaria Segoni (CERN)
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

#include<string>
#include<map>

class RPCDetId;

class RPCMonitorDigi : public edm::EDAnalyzer {
   public:
	explicit RPCMonitorDigi( const edm::ParameterSet&);
	~RPCMonitorDigi();
	
	virtual void analyze( const edm::Event&, const edm::EventSetup& );

	virtual void beginJob(edm::EventSetup const&);
	virtual void endJob(void);
        void beginRun(const edm::Run& r, const edm::EventSetup& c);

	/// Booking of MonitoringElemnt for one RPCDetId (= roll)
	std::map<std::string, MonitorElement*> bookDetUnitME(RPCDetId & detId);	
	
	
	/// Booking of MonitoringElemnt at Wheel/Disk level
	std::map<std::string, MonitorElement*> bookRegionRing(int region, int ring);

      
	

   private:
	
	int counter;
	/// back-end interface
	DQMStore * dbe;
        MonitorElement * GlobalZYHitCoordinates;
        MonitorElement * GlobalZXHitCoordinates;
        MonitorElement * GlobalZPhiHitCoordinates;
        
	MonitorElement * ClusterSize_for_Barrel;
        MonitorElement * ClusterSize_for_EndcapForward;
        MonitorElement * ClusterSize_for_EndcapBackward;
	MonitorElement * ClusterSize_for_BarrelandEndcaps;

	MonitorElement * s1;

	MonitorElement * NumberofClusters_for_Barrel;

	MonitorElement * SameBxDigisMe_;
        
	std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
        std::map<std::pair<int,int>, std::map<std::string, MonitorElement*> >  meWheelDisk;
	

	bool mergeRuns_;

	std::string nameInLog;
	bool saveRootFile;
	int  saveRootFileEventsInterval;
	std::string RootFileName;
	bool dqmshifter;
	bool dqmexpert;
	bool dqmsuperexpert;
	std::string GlobalHistogramsFolder;
	std::map<uint32_t,bool> foundHitsInChamber;
};

#endif
