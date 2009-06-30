#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

/** \class RPCMonitor
 *
 * Class for RPC Monitoring (strip id, cluster size).
 *
 *  $Date: 2008/12/15 16:38:08 $
 *  $Revision: 1.22 $
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
	/// DQM store 
	DQMStore * dbe;
        //MonitorElement * GlobalZYHitCoordinates;
        //MonitorElement * GlobalZXHitCoordinates;
        //MonitorElement * GlobalZPhiHitCoordinates;
        
	MonitorElement * NumberOfDigis_for_Barrel;
	MonitorElement * NumberOfDigis_for_EndcapPositive;
	MonitorElement * NumberOfDigis_for_EndcapNegative;

	MonitorElement * NumberOfClusters_for_Barrel;
	MonitorElement * NumberOfClusters_for_EndcapPositive;
	MonitorElement * NumberOfClusters_for_EndcapNegative;

	MonitorElement * ClusterSize_for_Barrel;
        MonitorElement * ClusterSize_for_EndcapPositive;
        MonitorElement * ClusterSize_for_EndcapNegative;
	
	MonitorElement * ClusterSize_for_BarrelandEndcaps;
	MonitorElement * BarrelNumberOfDigis;
	MonitorElement * BarrelOccupancy;
	MonitorElement * EndcapPositiveOccupancy;
	MonitorElement * EndcapNegativeOccupancy;
	MonitorElement * RPCEvents;

	MonitorElement * 	SameBxDigisMeBarrel_;
	MonitorElement * 	SameBxDigisMeEndcapPositive_;
	MonitorElement * 	SameBxDigisMeEndcapNegative_ ;

	std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
        std::map<std::pair<int,int>, std::map<std::string, MonitorElement*> >  meWheelDisk;
	
	std::string RPCDataLabel;
	std::string digiLabel;

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
