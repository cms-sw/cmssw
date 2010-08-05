#include <set>
#include <sstream>
#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Candidate/interface/Candidate.h"
///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
//Tracking Tools
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//FW Core
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Reco Muon
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"


const int RPCMonitorDigi::recHitTypesNum = 2; 
const std::string RPCMonitorDigi::regionNames_[3] =  {"Endcap-", "Barrel", "Endcap+"};
const std::string RPCMonitorDigi::recHitTypes_[recHitTypesNum] =  {"Noise", "Muon"};

RPCMonitorDigi::RPCMonitorDigi( const edm::ParameterSet& pset ):counter(0){

  saveRootFile  = pset.getUntrackedParameter<bool>("DigiDQMSaveRootFile", false); 

  onlyNoise_ = pset.getUntrackedParameter<bool>("Noise", false); 

  useMuonDigis_=  pset.getUntrackedParameter<bool>("Muon", true);

  muPtCut_  = pset.getUntrackedParameter<double>("MuonPtCut", 3.0); 

  muEtaCut_ = pset.getUntrackedParameter<double>("MuonEtaCut", 1.6); 
 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameDigi", "RPCMonitor.root"); 

  subsystemFolder_ = pset.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  globalFolder_ = pset.getUntrackedParameter<std::string>("GlobalFolder", "SummaryHistograms");

  //  rpcRecHitLabel = pset.getUntrackedParameter<std::string>("RecHitLabel","rpcRecHitLabel");
  muonLabel_ = pset.getUntrackedParameter<std::string>("MuonLabel","muons");
}

RPCMonitorDigi::~RPCMonitorDigi(){}


void RPCMonitorDigi::beginJob(){}


void RPCMonitorDigi::beginRun(const edm::Run& r, const edm::EventSetup& iSetup){

  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Begin Run " ;
  
  /// get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  //Book Summary histograms
  for(int i = 0 ; i< RPCMonitorDigi::recHitTypesNum; i++){
    this->bookSummaryHisto(RPCMonitorDigi::recHitTypes_[i]);
  }


  if(useMuonDigis_ ){
    std::string currentFolder = subsystemFolder_ +"/Muon/"+ globalFolder_;
    dbe->setCurrentFolder(currentFolder); 
    
    NumberOfMuonEta_ = dbe->get(currentFolder+"/NumberOfMuonEta");
    if(NumberOfMuonEta_) dbe->removeElement(NumberOfMuonEta_->getName());
    NumberOfMuonEta_ = dbe->book1D("NumberOfMuonEta", "Muons vs Eta", 32, -1.6, 1.6);
    
    RPCRecHitMuonEta_ = dbe->get(currentFolder+"/RPCRecHitMuonEta");
    if(RPCRecHitMuonEta_) dbe->removeElement(RPCRecHitMuonEta_->getName());
    RPCRecHitMuonEta_ = dbe->book2D("RPCRecHitMuonEta", "Number Of RecHit per Muons vs Eta", 32, -1.6, 1.6, 7, 0.5, 7.5);
  }
   
  RPCEvents_= dbe->get(subsystemFolder_ +"/RPCEvents");
  if(RPCEvents_) dbe->removeElement(RPCEvents_->getName());
  dbe->cd();
  dbe->setCurrentFolder(subsystemFolder_);
  RPCEvents_ = dbe->bookInt("RPCEvents");
  
  //Set DCS flag to default value
  dcs_ = true;


  dbe->cd();
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  //loop on geometry to book all MEs
  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Booking histograms per roll. " ;
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	
	//booking all histograms
	RPCGeomServ rpcsrv(rpcId);
	std::string nameRoll = rpcsrv.name();
	if(useMuonDigis_) meMuonCollection[(uint32_t)rpcId] = bookDetUnitME(rpcId,iSetup, "Muon");
	meNoiseCollection[(uint32_t)rpcId] = bookDetUnitME(rpcId,iSetup, "Noise");
	int ring;
	if(rpcId.region() == 0) 
	  ring = rpcId.ring();
	else 
	  ring = rpcId.region()*rpcId.station();
	
	//book wheel/disk histos
	std::pair<int,int> regionRing(region,ring);
	std::map<std::pair<int,int>, std::map<std::string,MonitorElement*> >::iterator meRingItr = meMuonWheelDisk.find(regionRing);
	if (useMuonDigis_ && ( meRingItr == meMuonWheelDisk.end() || meMuonWheelDisk.size()==0))  meMuonWheelDisk[regionRing]=bookRegionRing(region,ring, "Muon");
	meRingItr = meNoiseWheelDisk.find(regionRing);
	if (meRingItr == meNoiseWheelDisk.end() || meNoiseWheelDisk.size()==0)  meNoiseWheelDisk[regionRing]=bookRegionRing(region,ring, "Noise");
      
      }
    }
  }//end loop on geometry to book all MEs
}

void RPCMonitorDigi::endJob(void){
  if(saveRootFile) dbe->save(RootFileName); 
  dbe = 0;
}

void RPCMonitorDigi::analyze(const edm::Event& event,const edm::EventSetup& setup ){
  
  //Check HV status
  this->makeDcsInfo(event);
  if( !dcs_){
    edm::LogWarning ("rpcmonitordigi") <<"[RPCMonitorDigi]: DCS bit OFF" ;  
    return;//if RPC not ON there's no need to continue
  }

  counter++;
  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Beginning analyzing event " << counter;
  RPCEvents_->Fill(counter);

 
  //Muons
  edm::Handle<reco::CandidateView> muonCands;
  event.getByLabel(muonLabel_, muonCands);
  
  rechitmuon_.clear();
  if(muonCands.isValid() && useMuonDigis_){
    
    int nStaMuons = muonCands->size();
    
    for( int i = 0; i < nStaMuons; i++ ) {
      
      const reco::Candidate & goodMuon = (*muonCands)[i];
      const reco::Muon * muCand = dynamic_cast<const reco::Muon*>(&goodMuon);
      
      if(muCand->pt() < muPtCut_  ||  fabs(muCand->eta())>muPtCut_) continue;
      float muEta = muCand->eta();
      NumberOfMuonEta_->Fill(muEta);
      
      reco::Track muTrack = (*(muCand->outerTrack()));
      std::vector<TrackingRecHitRef > rpcTrackRecHits;
      trackingRecHit_iterator it= muTrack.recHitsBegin();
      unsigned int i = 0;
      //loop on mu rechits
      int numRPCRecHits =0;
      while (it!= muTrack.recHitsEnd() && i< muTrack.recHitsSize()){
	if ((*it)->isValid ()) { //select only valid rechits
	  int muSubDetId = (*it)->geographicalId().subdetId();
	  //select only RPC rechits associated to a muon
	  if(muSubDetId == MuonSubdetId::RPC)  {
	    numRPCRecHits++;
	    TrackingRecHit * tkRecHit = (*it)->clone();
	    RPCRecHit* rpcRecHit = dynamic_cast<RPCRecHit*>(tkRecHit);
	    rechitmuon_.push_back(*rpcRecHit);
	  } 
	}
	it++;
	i++;
      }// end loop on mu rechits
      
      if(numRPCRecHits) RPCRecHitMuonEta_->Fill(muEta,numRPCRecHits);
      
    }
  }else{
    edm::LogError ("rpcmonitordigi") <<"[RPCMonitorDigi]: Muons - Product not valid for event" << counter;
  }
  
 //RecHits
  edm::Handle<RPCRecHitCollection> rpcHits;
  event.getByType(rpcHits);
  
  rechitNOmuon_.clear();
  if(rpcHits.isValid()){
    //Get RPC rec hits NOT associated to a muon
    RPCRecHitCollection::const_iterator rpcRecHitIter;
    std::vector<RPCRecHit>::const_iterator muonRecHitIter;
    
    for (rpcRecHitIter = rpcHits->begin(); rpcRecHitIter != rpcHits->end() ; rpcRecHitIter++) {
      RPCRecHit rpcRecHit = (*rpcRecHitIter);
      if(!onlyNoise_ || rechitmuon_.size() == 0) {
	rechitNOmuon_.push_back(rpcRecHit);
      } else {
	bool isMuon = false;
	muonRecHitIter != rechitmuon_.begin();
	
	while (muonRecHitIter != rechitmuon_.end() && !isMuon) {
	  RPCRecHit muonRecHit = (*muonRecHitIter);
	  if(rpcRecHit == muonRecHit) isMuon = true;	
	}
	if(!isMuon ) rechitNOmuon_.push_back(rpcRecHit);
      }
    }
  }else {
    edm::LogError ("rpcmonitordigi") <<"[RPCMonitorDigi]: RPCRecHits - Product not valid for event" << counter;
  }

  if(useMuonDigis_) this->performSourceOperation(rechitmuon_, "Muon");
  this->performSourceOperation(rechitNOmuon_, "Noise");
}


void RPCMonitorDigi::performSourceOperation(std::vector<RPCRecHit> & recHits, std::string recHittype){


  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Performing DQM source operations for "<< recHittype<< "\n \t Found  "<<recHits.size() <<" recHits";
  
  if(recHits.size()==0) return;

  std::vector<RPCRecHit>::const_iterator recHitIter;

  std::map<uint32_t, std::map<std::string, MonitorElement*> >  meCollection;
  std::map<std::pair<int,int>, std::map<std::string, MonitorElement*> >  meWheelDisk;

  if(recHittype == "Muon") {
    meCollection =  meMuonCollection;
    meWheelDisk = meMuonWheelDisk;
  }else if (recHittype == "Noise" ){
    meCollection =  meNoiseCollection;
    meWheelDisk = meNoiseWheelDisk;
  }else{
    edm::LogWarning("rpcmonitordigi")<<"[RPCMonitorDigi]: RecHit type not valid.";
    return;
  }

  int numberOfDigis[3]= {0,0,0};
  int numberOfRecHits[3] ={0,0,0};

  std:: map<RPCDetId, int> recHitMap;
  std:: map<RPCDetId, int> digiMap;
  std:: map<RPCDetId, std::set<int> > bxMap;

  //Loop on recHits
  for(recHitIter = recHits.begin(); recHitIter != recHits.end(); recHitIter++){
    RPCRecHit recHit = (*recHitIter);
    int bx = recHit.BunchX();
   
    int clusterSize = (int)recHit.clusterSize();
    int firstStrip = recHit.firstClusterStrip();
    int lastStrip = clusterSize + firstStrip;

    RPCDetId detId = recHit.rpcId();
    int id=detId();

    recHitMap[detId]++;       
    digiMap[detId]+=clusterSize;   
    bxMap[detId].insert(bx);

    RPCGeomServ geoServ(detId);

    //Roll name
    std::string nameRoll = geoServ.name();

     //get roll number
    rpcdqm::utils rpcUtils;
    int nr = rpcUtils.detId2RollNr(detId);

    int region=(int)detId.region();
    int ring;
    std::string ringType;
    if(region == 0) {
      ringType = "Wheel";  
      ring = (int)detId.ring();
      numberOfDigis[B]+=clusterSize;
      numberOfRecHits[B]++;
    }else if (region ==1){
      ringType =  "Disk";
      ring = region*detId.station();
      numberOfDigis[EP]+=clusterSize; 
      numberOfRecHits[EP]++;
    }else {
      ringType =  "Disk";
      ring = region*(int)detId.station();
      numberOfDigis[EM]+=clusterSize; 
      numberOfRecHits[EM]++;
    }

    //get MEs corresponding to present detId  
    std::map<std::string, MonitorElement*> meMap=meCollection[id]; 
    if(meMap.size()==0) continue; 

    //get wheel/disk MEs
    std::pair<int,int> regionRing(region,ring);
    std::map<std::string, MonitorElement*> meRingMap=meWheelDisk[regionRing];
    if(meRingMap.size()==0) continue;

    //Fill histograms
    std::stringstream os;

    // ###################### Occupancy #################################

    os.str("");
    os<<"Occupancy_"<<nameRoll;
    if(meMap[os.str()]) {
      for(int s=firstStrip; s<= lastStrip; s++){
	meMap[os.str()]->Fill(s);
      }
    }

    os.str("");
    os<<"Occupancy_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
    if(meMap[os.str()]){ 
      for(int s=firstStrip; s<= lastStrip; s++){//Loop on digis
	if(detId.region() ==0)	meMap[os.str()]->Fill(s, nr);
	else meMap[os.str()]->Fill(s + 32*(detId.roll()-1),  geoServ.segment()+ ((detId.ring() -2)*6));
      }
    }

    os.str("");
    os<<"1DOccupancy_"<<ringType<<"_"<<ring;
    std::string meId = os.str();
    if( meRingMap[meId])  meRingMap[meId]->Fill(detId.sector(), clusterSize);
    
    os.str("");
    os<<"Occupancy_Roll_vs_Sector_"<<ringType<<"_"<<ring;       
    if (meRingMap[os.str()]) {
      meRingMap[os.str()]->Fill(detId.sector(), nr, clusterSize);
    }
 
    os.str("");
    os<<"Occupancy_Ring_vs_Segment_"<<ringType<<"_"<<ring;   
    if (meRingMap[os.str()]) {
      meRingMap[os.str()]->Fill( geoServ.segment(), (detId.ring()-1)*3-detId.roll()+1,clusterSize );
    }


    // ######################  BX  ######################################

    os.str("");
    os<<"BXDistribution_"<<nameRoll;
    if(meMap[os.str()]) meMap[os.str()]->Fill(bx);

    os.str("");
    os<<"BxDistribution_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
    if(meMap[os.str()]) meMap[os.str()]->Fill(bx);
    
    os.str("");
    os<<"BxDistribution_"<<ringType<<"_"<<ring;
    if(meRingMap[os.str()])      meRingMap[os.str()]->Fill(bx);

  
    // ######################  Cluster ##################################

    os.str("");
    os<<"ClusterSize_"<<ringType<<"_"<<ring;
    if(meRingMap[os.str()]) meRingMap[os.str()] -> Fill(clusterSize);
   
    os.str("");
    os<<"ClusterSize_"<<nameRoll;
    if(meMap[os.str()]) meMap[os.str()]->Fill(clusterSize);
 
    // ######################  Global  ##################################
    if (region != 0){
      std::cout<<detId.sector()<<std::endl;
      std::cout<<ring<<std::endl;
      std::cout<<clusterSize<<std::endl;
    }

    if( Occupancy_[region+1])  Occupancy_[region+1]->Fill(detId.sector(), ring , clusterSize); 

    if( ClusterSize_[region+1])  ClusterSize_[region+1]->Fill(clusterSize); 
  
  }//end loop on recHits

  

  //Fill histograms for "per event"
  
  if( NumberOfClusters_[B])  NumberOfClusters_[B]->Fill(numberOfRecHits[B]); 
  if( NumberOfClusters_[EP])  NumberOfClusters_[EP]->Fill(numberOfRecHits[EP]); 
  if( NumberOfClusters_[EM])  NumberOfClusters_[EM]->Fill(numberOfRecHits[EM]); 
  
  if( NumberOfDigis_[B])  NumberOfClusters_[B]->Fill(numberOfDigis[B]); 
  if( NumberOfDigis_[EP])  NumberOfClusters_[EP]->Fill(numberOfDigis[EP]); 
  if( NumberOfDigis_[EM])  NumberOfClusters_[EM]->Fill(numberOfDigis[EM]); 
   
  if( bxMap.size()!= digiMap.size() || bxMap.size()!= recHitMap.size()) {
    edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Roll maps do NOT have the same size!";
  } else {
        for(std::map<RPCDetId, int>::const_iterator it = recHitMap.begin() ; it != recHitMap.begin(); it++){

	  int clusterSize = (*it).second;
	  RPCDetId detId = (*it).first;
	  RPCGeomServ geoServ(detId);
	  std::string nameRoll = geoServ.name();

	  std::map<std::string, MonitorElement*> meMap=meCollection[detId]; 
	  if(meMap.size()==0) continue; 
	 	  
	  std::stringstream os;
	  os.str("");
	  os<<"BXWithData_"<<nameRoll;
	  if(meMap[os.str()]) meMap[os.str()]->Fill(bxMap[detId].size());
	  
	  os.str("");
	  os<<"NumberOfClusters_"<<nameRoll;
	  if(meMap[os.str()]) meMap[os.str()]->Fill(clusterSize);
	 
	  os.str("");
	  os<<"Multiplicity_"<<nameRoll;
	  if(meMap[os.str()]) meMap[os.str()]->Fill(digiMap[detId]);   

	  if( NumberOfDigis_[detId.region() +1 ])  NumberOfDigis_[detId.region() +1]->Fill(digiMap[detId]); 
	}

  }
}


void  RPCMonitorDigi::makeDcsInfo(const edm::Event& e) {

  edm::Handle<DcsStatusCollection> dcsStatus;

  if ( ! e.getByLabel("scalersRawToDigi", dcsStatus) ){
    dcs_ = true;
    return;
  }
  
  if ( ! dcsStatus.isValid() ) 
  {
    edm::LogWarning("RPCDcsInfo") << "scalersRawToDigi not found" ;
    dcs_ = true; // info not available: set to true
    return;
  }
    

  for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
                            dcsStatusItr != dcsStatus->end(); ++dcsStatusItr){

      if (!dcsStatusItr->ready(DcsStatus::RPC)) dcs_=false;
      
  }
      
  return ;
}


