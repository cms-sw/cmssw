#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include<string>
#include<map>

class RPCMonitorDigi : public edm::EDAnalyzer {
   public:
	explicit RPCMonitorDigi( const edm::ParameterSet&);
	~RPCMonitorDigi();
	
	virtual void analyze( const edm::Event&, const edm::EventSetup& );

	virtual void beginJob();
	virtual void endJob(void);
        void beginRun(const edm::Run& r, const edm::EventSetup& c);

	/// Booking of MonitoringElemnt for one RPCDetId (= roll)
	std::map<std::string, MonitorElement*> bookDetUnitME(RPCDetId& , const edm::EventSetup&, std::string );		
	
	/// Booking of MonitoringElemnt at Wheel/Disk level
	std::map<std::string, MonitorElement*> bookRegionRing(int region, int ring, std::string );

      

   private:

	enum detectorRegions{EM = 0, B = 1, EP= 2, ALL=3};
	bool	useMuonDigis_;
	void performSourceOperation(std::vector<RPCRecHit>&, std::string );
	void bookSummaryHisto(std::string );
	void makeDcsInfo(const edm::Event& ) ;
       	int stripsInRoll(RPCDetId & ,const edm::EventSetup& );

	static const std::string recHitTypes_[2]; 
	static const std::string regionNames_[3];
	static const int recHitTypesNum;

	bool onlyNoise_;
	int counter;
	/// DQM store 
	DQMStore * dbe;
	bool dcs_;
	float muPtCut_, muEtaCut_;
	MonitorElement * NumberOfDigis_[3];
	MonitorElement * NumberOfClusters_[3];
	MonitorElement * ClusterSize_[3];
      	MonitorElement * Occupancy_[3];
      	MonitorElement * NumberOfMuonEta_ ;
      	MonitorElement * RPCRecHitMuonEta_ ;
	MonitorElement * RPCEvents_;


	std::vector< RPCRecHit > rechitmuon_;
	std::vector< RPCRecHit > rechitNOmuon_;

	std::map<uint32_t, std::map<std::string, MonitorElement*> >  meMuonCollection;
        std::map<std::pair<int,int>, std::map<std::string, MonitorElement*> >  meMuonWheelDisk;
	
	std::map<uint32_t, std::map<std::string, MonitorElement*> >  meNoiseCollection;
        std::map<std::pair<int,int>, std::map<std::string, MonitorElement*> >  meNoiseWheelDisk;

	std::string muonLabel_;

	std::string globalFolder_;
	std::string subsystemFolder_;

	bool saveRootFile;
	std::string RootFileName;

	edm::ESHandle<RPCGeometry> rpcGeo;

};

#endif
