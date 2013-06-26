#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include<string>
#include<map>

class RPCMonitorDigi : public edm::EDAnalyzer {
   public:
	explicit RPCMonitorDigi( const edm::ParameterSet&);
	~RPCMonitorDigi();
	
	virtual void analyze( const edm::Event&, const edm::EventSetup& );

	virtual void beginJob();
	
	void endLuminosityBlock(edm::LuminosityBlock const& , edm::EventSetup const& );

	virtual void endJob(void);
        void beginRun(const edm::Run& r, const edm::EventSetup& c);

	/// Booking of MonitoringElement for one RPCDetId (= roll)
	//	std::map<std::string, MonitorElement*> bookRollME(RPCDetId& , const edm::EventSetup&, std::string );		
	void bookRollME(RPCDetId& , const edm::EventSetup&, const std::string &, std::map<std::string, MonitorElement*> &);

	/// Booking of MonitoringElement at Sector/Ring level
	//	std::map<std::string, MonitorElement*> bookSectorRingME(std::string);
	void bookSectorRingME(const std::string&, std::map<std::string, MonitorElement*> &);

	/// Booking of MonitoringElemnt at Wheel/Disk level
	//	std::map<std::string, MonitorElement*> bookWheelDiskME(std::string );
	void bookWheelDiskME(const std::string &, std::map<std::string, MonitorElement*> &);



	/// Booking of MonitoringElemnt at region (Barrel/Endcap) level
	//	std::map<std::string, MonitorElement*> bookRegionME(std::string );
      void bookRegionME(const std::string &, std::map<std::string, MonitorElement*> &);

   private:

	enum detectorRegions{EM = 0, B = 1, EP= 2, ALL=3};

	bool useMuonDigis_;

	void performSourceOperation(std::map < RPCDetId , std::vector<RPCRecHit> > &, std::string );
	void makeDcsInfo(const edm::Event& ) ;
	int stripsInRoll(RPCDetId & ,const edm::EventSetup& );

	static const std::string regionNames_[3];
	std::string muonFolder_;
	std::string noiseFolder_;
	int counter;

	/// DQM store 
	DQMStore * dbe;
	bool dcs_;
	float muPtCut_, muEtaCut_;
	bool useRollInfo_;
 	MonitorElement * noiseRPCEvents_ ;
	MonitorElement * muonRPCEvents_ ;

	MonitorElement * NumberOfRecHitMuon_;
	MonitorElement * NumberOfMuon_;

	int numberOfDisks_, numberOfInnerRings_;
	//	int muonCounter_, noiseCounter_;

	std::map< std::string, std::map<std::string, MonitorElement*> >   meMuonCollection;
	std::map<std::string, MonitorElement*>  wheelDiskMuonCollection;
	std::map<std::string, MonitorElement*>  regionMuonCollection;
	std::map<std::string, MonitorElement*> sectorRingMuonCollection;
	
	std::map<std::string, std::map<std::string, MonitorElement*> > meNoiseCollection;
	std::map<std::string, MonitorElement*> wheelDiskNoiseCollection;
	std::map<std::string, MonitorElement*>  regionNoiseCollection;
	std::map<std::string, MonitorElement*> sectorRingNoiseCollection;
   
	std::string globalFolder_;
	std::string subsystemFolder_;

	bool saveRootFile;
	std::string RootFileName;

	edm::InputTag rpcRecHitLabel_;
	edm::InputTag muonLabel_;
};

#endif
