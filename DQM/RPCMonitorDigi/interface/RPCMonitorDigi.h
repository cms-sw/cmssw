#ifndef RPCMonitorDigi_h
#define RPCMonitorDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//DQMServices
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

///Data Format
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include<string>
#include<map>

class RPCMonitorDigi : public DQMEDAnalyzer {
   public:
	explicit RPCMonitorDigi( const edm::ParameterSet&);
	~RPCMonitorDigi();
	

 protected:

	virtual void analyze( const edm::Event&, const edm::EventSetup& );
	void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
	/// Booking of MonitoringElement for one RPCDetId (= roll)
	void bookRollME(DQMStore::IBooker &,RPCDetId& , const edm::EventSetup&, const std::string &, std::map<std::string, MonitorElement*> &);
	/// Booking of MonitoringElement at Sector/Ring level
	void bookSectorRingME(DQMStore::IBooker &,const std::string&, std::map<std::string, MonitorElement*> &);
	/// Booking of MonitoringElemnt at Wheel/Disk level
	void bookWheelDiskME(DQMStore::IBooker &,const std::string &, std::map<std::string, MonitorElement*> &);
	/// Booking of MonitoringElemnt at region (Barrel/Endcap) level
	void bookRegionME(DQMStore::IBooker &,const std::string &, std::map<std::string, MonitorElement*> &);

   private:

	enum detectorRegions{EM = 0, B = 1, EP= 2, ALL=3};

	bool useMuonDigis_;

	void performSourceOperation(std::map < RPCDetId , std::vector<RPCRecHit> > &, std::string );
	int stripsInRoll(RPCDetId & ,const edm::EventSetup& );

	static const std::string regionNames_[3];
	std::string muonFolder_;
	std::string noiseFolder_;
	int counter;

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

	edm::EDGetTokenT<reco::CandidateView> muonLabel_;
	edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitLabel_;
	edm::EDGetTokenT<DcsStatusCollection> scalersRawToDigiLabel_;



};

#endif
