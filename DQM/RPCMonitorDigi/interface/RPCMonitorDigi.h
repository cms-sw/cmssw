#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring (strip id, cluster size).
 *
 *  $Date: 2008/05/04 13:55:19 $
 *  $Revision: 1.16 $
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
	std::map<std::string, MonitorElement*> bookDetUnitME(RPCDetId& , const edm::EventSetup&);	
	
	
	/// Booking of MonitoringElemnt at Wheel/Disk level
	std::map<std::string, MonitorElement*> bookRegionRing(int region, int ring);

      
	

   private:
	
	int stripsInRoll(RPCDetId & ,const edm::EventSetup& );
	int counter;
	/// back-end interface
	DQMStore * dbe;
        //MonitorElement * GlobalZYHitCoordinates;
        //MonitorElement * GlobalZXHitCoordinates;
        //MonitorElement * GlobalZPhiHitCoordinates;
        
	MonitorElement * NumberOfDigis_for_Barrel;
	MonitorElement * NumberOfClusters_for_Barrel;
	MonitorElement * ClusterSize_for_Barrel;
        MonitorElement * ClusterSize_for_EndcapForward;
        MonitorElement * ClusterSize_for_EndcapBackward;
	MonitorElement * ClusterSize_for_BarrelandEndcaps;

	MonitorElement * BarrelOccupancy;

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
