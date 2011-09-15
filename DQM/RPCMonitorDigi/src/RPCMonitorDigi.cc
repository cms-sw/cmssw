#include <set>
#include <sstream>
#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
///Data Format
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/MuonReco/interface/Muon.h"
///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//Tracking Tools
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//FW Core
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Reco Muon
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

const std::string RPCMonitorDigi::regionNames_[3] =  {"Endcap-", "Barrel", "Endcap+"};

RPCMonitorDigi::RPCMonitorDigi( const edm::ParameterSet& pset ):counter(0){

  saveRootFile  = pset.getUntrackedParameter<bool>("SaveRootFile", false); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileName", "RPCMonitorDigiDQM.root"); 

  useMuonDigis_=  pset.getUntrackedParameter<bool>("UseMuon", true);
  useRollInfo_=  pset.getUntrackedParameter<bool>("UseRollInfo", false);
  muonLabel_ = pset.getParameter<edm::InputTag>("MuonLabel");
  muPtCut_  = pset.getUntrackedParameter<double>("MuonPtCut", 3.0); 
  muEtaCut_ = pset.getUntrackedParameter<double>("MuonEtaCut", 1.9); 
 
  subsystemFolder_ = pset.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  globalFolder_ = pset.getUntrackedParameter<std::string>("GlobalFolder", "SummaryHistograms");

  rpcRecHitLabel_ = pset.getParameter<edm::InputTag>("RecHitLabel");

  numberOfDisks_ = pset.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);
  numberOfInnerRings_ = pset.getUntrackedParameter<int>("NumberOfInnermostEndcapRings", 2);

  noiseFolder_ = pset.getUntrackedParameter<std::string>("NoiseFolder", "Noise");
  muonFolder_ = pset.getUntrackedParameter<std::string>("MuonFolder", "Muon");

}

RPCMonitorDigi::~RPCMonitorDigi(){}
void RPCMonitorDigi::beginJob(){}

void RPCMonitorDigi::beginRun(const edm::Run& r, const edm::EventSetup& iSetup){

  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Begin Run " ;
  
  /// get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();

  //Book 

  //regionNoiseCollection = this->bookRegionME(noiseFolder_);
  //sectorRingNoiseCollection = this->bookSectorRingME(noiseFolder_);
  //wheelDiskNoiseCollection = this->bookWheelDiskME(noiseFolder_);
  
  this->bookRegionME(noiseFolder_, regionNoiseCollection);
  this->bookSectorRingME(noiseFolder_, sectorRingNoiseCollection);
  this->bookWheelDiskME(noiseFolder_, wheelDiskNoiseCollection);



  std::string currentFolder = subsystemFolder_ +"/"+noiseFolder_;
  dbe->setCurrentFolder(currentFolder);
  
  noiseRPCEvents_= dbe->get(currentFolder +"/RPCEvents");
  if(noiseRPCEvents_) dbe->removeElement(noiseRPCEvents_->getName());
  noiseRPCEvents_ = dbe->book1D("RPCEvents","RPCEvents", 1, 0.5, 1.5);
  
  
  if(useMuonDigis_ ){
  //   regionMuonCollection = this->bookRegionME(muonFolder_);
//     sectorRingMuonCollection = this->bookSectorRingME(muonFolder_);
//     wheelDiskMuonCollection = this->bookWheelDiskME(muonFolder_);
    this->bookRegionME(muonFolder_, regionMuonCollection);
    this->bookSectorRingME(muonFolder_, sectorRingMuonCollection);
    this->bookWheelDiskME(muonFolder_, wheelDiskMuonCollection);
    
    currentFolder = subsystemFolder_ +"/"+muonFolder_;
    dbe->setCurrentFolder(currentFolder); 
   
    muonRPCEvents_= dbe->get(currentFolder +"/RPCEvents");
    if(muonRPCEvents_) dbe->removeElement(muonRPCEvents_->getName());
    muonRPCEvents_ =  dbe->book1D("RPCEvents", "RPCEvents", 1, 0.5, 1.5);

    NumberOfMuon_ = dbe->get(currentFolder+"/NumberOfMuons");
    if(NumberOfMuon_) dbe->removeElement(NumberOfMuon_->getName());
    NumberOfMuon_ = dbe->book1D("NumberOfMuons", "Number of Muons", 11, -0.5, 10.5);


    NumberOfRecHitMuon_ = dbe->get(currentFolder+"/NumberOfRPCRecHitsMuons");
    if(NumberOfRecHitMuon_) dbe->removeElement(NumberOfRecHitMuon_->getName());
    NumberOfRecHitMuon_ = dbe->book1D("NumberOfRecHitMuons", "Number of RPC RecHits per Muon", 8, -0.5, 7.5);
  }
   

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  //loop on geometry to book all MEs
  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Booking histograms per roll. " ;
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      if(useRollInfo_){
	for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	  RPCDetId rpcId = (*r)->id();
	  //booking all histograms
	  RPCGeomServ rpcsrv(rpcId);
	  std::string nameID = rpcsrv.name();
	  if(useMuonDigis_) bookRollME(rpcId ,iSetup, muonFolder_, meMuonCollection[nameID]);
	  bookRollME(rpcId, iSetup, noiseFolder_, meNoiseCollection[nameID]);
	  
	  // 	if(useMuonDigis_) meMuonCollection[(uint32_t)rpcId] = bookRollME(rpcId,iSetup,  muonFolder_);
	  // 	meNoiseCollection[(uint32_t)rpcId] = bookRollME(rpcId,iSetup,  noiseFolder_);
	}
      }else{
	RPCDetId rpcId = roles[0]->id(); //any roll would do - here I just take the first one
	RPCGeomServ rpcsrv(rpcId);
	std::string nameID = rpcsrv.chambername();
	if(useMuonDigis_) bookRollME(rpcId,iSetup, muonFolder_, meMuonCollection[nameID]);
	bookRollME(rpcId, iSetup, noiseFolder_, meNoiseCollection[nameID]);
	
      }
    }
  }//end loop on geometry to book all MEs


  //Clear flags;
  dcs_ = true;
}

void RPCMonitorDigi::endJob(void){
  if(saveRootFile) dbe->save(RootFileName); 
  dbe = 0;
}

void RPCMonitorDigi::endLuminosityBlock(edm::LuminosityBlock const& L, edm::EventSetup const&  E){}


void RPCMonitorDigi::analyze(const edm::Event& event,const edm::EventSetup& setup ){
  dcs_ = true;
  //Check HV status
  this->makeDcsInfo(event);
  if( !dcs_){
    edm::LogWarning ("rpcmonitordigi") <<"[RPCMonitorDigi]: DCS bit OFF" ;  
    return;//if RPC not ON there's no need to continue
  }

  counter++;
  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Beginning analyzing event " << counter;
 
  //Muons
  edm::Handle<reco::CandidateView> muonCands;
  event.getByLabel(muonLabel_, muonCands);
  std::map<RPCDetId  , std::vector<RPCRecHit> > rechitMuon;

  int  numMuons = 0;
  int  numRPCRecHit = 0 ;

  if(muonCands.isValid()){

    int nStaMuons = muonCands->size();
    
    for( int i = 0; i < nStaMuons; i++ ) {
      
      const reco::Candidate & goodMuon = (*muonCands)[i];
      const reco::Muon * muCand = dynamic_cast<const reco::Muon*>(&goodMuon);
    
      if(!muCand->isGlobalMuon())continue;
      if(muCand->pt() < muPtCut_  ||  fabs(muCand->eta())>muEtaCut_) continue;
      numMuons++;
      reco::Track muTrack = (*(muCand->outerTrack()));
      std::vector<TrackingRecHitRef > rpcTrackRecHits;
      //loop on mu rechits
      for ( trackingRecHit_iterator it= muTrack.recHitsBegin(); it !=  muTrack.recHitsEnd() ; it++) {
	if (!(*it)->isValid ())continue;
	int muSubDetId = (*it)->geographicalId().subdetId();
	if(muSubDetId == MuonSubdetId::RPC)  {
	  numRPCRecHit ++;
	  TrackingRecHit * tkRecHit = (*it)->clone();
	  RPCRecHit* rpcRecHit = dynamic_cast<RPCRecHit*>(tkRecHit);
	  int detId = (int)rpcRecHit->rpcId();
	  if(rechitMuon.find(detId) == rechitMuon.end() || rechitMuon[detId].size() == 0){
	    std::vector<RPCRecHit>  myVect(1,*rpcRecHit );	  
	    rechitMuon[detId]= myVect;
	  }else {
	    rechitMuon[detId].push_back(*rpcRecHit);
	  }
	}
      }// end loop on mu rechits
    
    }

    if( NumberOfMuon_)  NumberOfMuon_->Fill(numMuons);
    if( NumberOfRecHitMuon_)  NumberOfRecHitMuon_->Fill( numRPCRecHit);
    
  }else{
    edm::LogError ("rpcmonitordigi") <<"[RPCMonitorDigi]: Muons - Product not valid for event" << counter;
  }
  
 //RecHits
  edm::Handle<RPCRecHitCollection> rpcHits;
  event.getByLabel( rpcRecHitLabel_ , rpcHits);
  std::map<RPCDetId  , std::vector<RPCRecHit> > rechitNoise;

  
  if(rpcHits.isValid()){
   
    //    RPC rec hits NOT associated to a muon
    RPCRecHitCollection::const_iterator rpcRecHitIter;
    std::vector<RPCRecHit>::const_iterator muonRecHitIter;
    
    for (rpcRecHitIter = rpcHits->begin(); rpcRecHitIter != rpcHits->end() ; rpcRecHitIter++) {
      RPCRecHit rpcRecHit = (*rpcRecHitIter);
      int detId = (int)rpcRecHit.rpcId();
      if(rechitNoise.find(detId) == rechitNoise.end() || rechitNoise[detId].size() == 0){
	std::vector<RPCRecHit>  myVect(1,rpcRecHit );
	rechitNoise[detId]= myVect;
      }else {
	rechitNoise[detId].push_back(rpcRecHit);
      }
    }
  }else{
    edm::LogError ("rpcmonitordigi") <<"[RPCMonitorDigi]: RPCRecHits - Product not valid for event" << counter;
  }

 
  if( useMuonDigis_ && muonRPCEvents_ != 0 )  muonRPCEvents_->Fill(1);
  if( noiseRPCEvents_ != 0)  noiseRPCEvents_->Fill(1);

  if(useMuonDigis_ ) this->performSourceOperation(rechitMuon, muonFolder_);
  this->performSourceOperation(rechitNoise, noiseFolder_);
}


void RPCMonitorDigi::performSourceOperation(  std::map<RPCDetId , std::vector<RPCRecHit> > & recHitMap, std::string recHittype){

  edm::LogInfo ("rpcmonitordigi") <<"[RPCMonitorDigi]: Performing DQM source operations for "; 
  
  if(recHitMap.size()==0) return;

  std::map<std::string, std::map<std::string, MonitorElement*> >  meRollCollection ;
  std::map<std::string, MonitorElement*>   meWheelDisk ;
  std::map<std::string, MonitorElement*>   meRegion ;
  std::map<std::string, MonitorElement*>   meSectorRing;  

  if(recHittype == muonFolder_ ) {
    meRollCollection = meMuonCollection;
    meWheelDisk =  wheelDiskMuonCollection;
    meRegion =  regionMuonCollection;
    meSectorRing =  sectorRingMuonCollection;
  }else if(recHittype == noiseFolder_ ){
    meRollCollection =  meNoiseCollection;
    meWheelDisk =  wheelDiskNoiseCollection;
    meRegion =  regionNoiseCollection;
    meSectorRing =  sectorRingNoiseCollection;
  }else{
    edm::LogWarning("rpcmonitordigi")<<"[RPCMonitorDigi]: RecHit type not valid.";
    return;
  }


  int totalNumberOfRecHits[3] ={ 0, 0, 0};
  std::stringstream os;

  //Loop on Rolls
  for ( std::map<RPCDetId , std::vector<RPCRecHit> >::const_iterator detIdIter = recHitMap.begin(); detIdIter !=  recHitMap.end() ;  detIdIter++){
    
    RPCDetId detId = (*detIdIter).first;
    // int id=detId();
    
    //get roll number
    rpcdqm::utils rpcUtils;
    int nr = rpcUtils.detId2RollNr(detId);
 
    
    RPCGeomServ geoServ(detId);
    std::string nameRoll = "";


    if(useRollInfo_) nameRoll = geoServ.name();
    else nameRoll = geoServ.chambername();

    int region=(int)detId.region();
    int wheelOrDiskNumber;
    std::string wheelOrDiskType;
    int ring = 0 ;
    int sector  = detId.sector();
    int layer = 0;
    int totalRolls = 3;
    int roll = detId.roll();
    if(region == 0) {
      wheelOrDiskType = "Wheel";  
      wheelOrDiskNumber = (int)detId.ring();
      int station = detId.station();
     
      if(station == 1){
	if(detId.layer() == 1){
	  layer = 1; //RB1in
	  totalRolls = 2;
	}else{
	  layer = 2; //RB1out
	  totalRolls = 2;
	}
	if(roll == 3) roll =2; // roll=3 is Forward
      }else if(station == 2){
      	if(detId.layer() == 1){
	  layer = 3; //RB2in
	  if( abs(wheelOrDiskNumber) ==2 && roll == 3) {
	    roll = 2; //W -2, +2 RB2in has only 2 rolls
	    totalRolls = 2;
	  }
       	}else{
	  layer = 4; //RB2out
	  if( abs(wheelOrDiskNumber) !=2 && roll == 3){
	    roll = 2;//W -1, 0, +1 RB2out has only 2 rolls
	    totalRolls = 2;
	  }
	}
      }else if (station == 3){
	layer = 5; //RB3
	totalRolls = 2;
	if(roll == 3) roll =2;
      }else{
	layer = 6; //RB4
	totalRolls = 2;
	if(roll == 3) roll =2;
      }

    }else {
      wheelOrDiskType =  "Disk";
      wheelOrDiskNumber = region*(int)detId.station();
      ring = detId.ring();
    }

    std::vector<RPCRecHit> recHits  = (*detIdIter).second;
    int numberOfRecHits = recHits.size();
    totalNumberOfRecHits[region + 1 ] +=  numberOfRecHits;

    std::set<int> bxSet ;
    int numDigi = 0;

    std::map<std::string, MonitorElement*>  meMap = meRollCollection[nameRoll];

    //Loop on recHits
    for(std::vector<RPCRecHit>::const_iterator recHitIter = recHits.begin(); recHitIter != recHits.end(); recHitIter++){
      RPCRecHit recHit = (*recHitIter);

      int bx = recHit.BunchX();
      bxSet.insert(bx); 
      int clusterSize = (int)recHit.clusterSize();
      numDigi +=  clusterSize ;
      int firstStrip = recHit.firstClusterStrip();
      int lastStrip = clusterSize + firstStrip - 1;
          
      // ###################### Roll Level  #################################
      
      os.str("");
      os<<"Occupancy_"<<nameRoll;
      if(meMap[os.str()]) {
	for(int s=firstStrip; s<= lastStrip; s++){
	  if(useRollInfo_) { meMap[os.str()]->Fill(s);}
	  else{ 
	    int nstrips =   meMap[os.str()]->getNbinsX()/totalRolls;
	    meMap[os.str()]->Fill(s + nstrips*(roll-1)); }
	}
      }
      
      os.str("");
      os<<"BXDistribution_"<<nameRoll;
      if(meMap[os.str()]) meMap[os.str()]->Fill(bx);

  
      os.str("");
      os<<"ClusterSize_"<<nameRoll;
      if(meMap[os.str()]) meMap[os.str()]->Fill(clusterSize);
 


      // ###################### Sector- Ring Level #################################


      os.str("");
      os<<"Occupancy_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Sector_"<<sector;
      if( meSectorRing[os.str()]){ 
	for(int s=firstStrip; s<= lastStrip; s++){//Loop on digis
	   meSectorRing[os.str()]->Fill(s, nr);
	}
      }

   //    os.str("");
//       os<<"BxDistribution_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Sector_"<<sector;
//       if( meSectorRing[os.str()])  meSectorRing[os.str()]->Fill(bx);

      os.str("");
      if(geoServ.segment() > 0 && geoServ.segment() < 19 ){ 
	os<<"Occupancy_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring<<"_CH01-CH18";
      }else if (geoServ.segment() > 18 ){
	os<<"Occupancy_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring<<"_CH19-CH36";
      }
     
      if( meSectorRing[os.str()]){ 
	for(int s=firstStrip; s<= lastStrip; s++){//Loop on digis
	   meSectorRing[os.str()]->Fill(s + 32*(detId.roll()-1),  geoServ.segment());
	}
      }

     //  os.str("");
//       os<<"BxDistribution_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring_"<<ring;
//       if( meSectorRing[os.str()])  meSectorRing[os.str()]->Fill(bx);

      
      // ###################### Wheel/Disk Level #########################‡‡‡
      if(region ==0){
	os.str("");
	os<<"1DOccupancy_Wheel_"<<wheelOrDiskNumber;
	if( meWheelDisk[os.str()]) meWheelDisk[os.str()]->Fill(sector, clusterSize);
	
	os.str("");
	os<<"Occupancy_Roll_vs_Sector_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber;       
	if (meWheelDisk[os.str()]) meWheelDisk[os.str()]->Fill(sector, nr, clusterSize);

      }else{
	os.str("");
	os<<"1DOccupancy_Ring_"<<ring;
	if ((meWheelDisk[os.str()])){
	  if (wheelOrDiskNumber > 0 ) meWheelDisk[os.str()]->Fill(wheelOrDiskNumber +3, clusterSize);
	  else meWheelDisk[os.str()]->Fill(wheelOrDiskNumber + 4, clusterSize);
	}

	os.str("");
	os<<"Occupancy_Ring_vs_Segment_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber;   
	if (meWheelDisk[os.str()]) meWheelDisk[os.str()]->Fill( geoServ.segment(), (ring-1)*3-detId.roll()+1,clusterSize );
      }

      os.str("");
      os<<"BxDistribution_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber;
      if(meWheelDisk[os.str()])  meWheelDisk[os.str()]->Fill(bx);
      
  
      os.str("");
      os<<"ClusterSize_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Layer"<<layer;
      if(meWheelDisk[os.str()]) meWheelDisk[os.str()] -> Fill(clusterSize);
 

      os.str("");
      os<<"ClusterSize_"<<wheelOrDiskType<<"_"<<wheelOrDiskNumber<<"_Ring"<<ring;
      if(meWheelDisk[os.str()]) meWheelDisk[os.str()] -> Fill(clusterSize);


    // ######################  Global  ##################################
 

      os.str("");
      os<<"ClusterSize_"<<RPCMonitorDigi::regionNames_[region +1];
      if(meRegion[os.str()]) meRegion[os.str()] -> Fill(clusterSize);

      os.str("");
      os<<"ClusterSize_";
      if(region == 0){
	os<<"Layer"<<layer;
      }else{
	os<<"Ring"<<ring;
      }
      if(meRegion[os.str()]) meRegion[os.str()] -> Fill(clusterSize);

      
    }//end loop on recHits
  
    os.str("");
    os<<"BXWithData_"<<nameRoll;
    if(meMap[os.str()]) meMap[os.str()]->Fill(bxSet.size());
    
    os.str("");
    os<<"NumberOfClusters_"<<nameRoll;
    if(meMap[os.str()]) meMap[os.str()]->Fill( numberOfRecHits);

    os.str("");
    os<<"Multiplicity_"<<RPCMonitorDigi::regionNames_[region +1];
    if(meRegion[os.str()]) meRegion[os.str()]->Fill(numDigi);

    os.str("");
    if(region==0) {
      os<<"Occupancy_for_Barrel";
      if(meRegion[os.str()]) meRegion[os.str()]->Fill(sector, wheelOrDiskNumber, numDigi);
    }else {
      os<<"Occupancy_for_Endcap";
      int xbin = wheelOrDiskNumber+3;
      if (region==-1) xbin = wheelOrDiskNumber+4;
      if(meRegion[os.str()]) meRegion[os.str()]->Fill(xbin,ring,numDigi);
    }

    os.str("");
    os<<"Multiplicity_"<<nameRoll;
    if(meMap[os.str()]) meMap[os.str()]->Fill(numDigi);   

  }//end loop on rolls

  for(int i = 0; i< 3; i++ ){
    os.str("");
    os<<"NumberOfClusters_"<<RPCMonitorDigi::regionNames_[i];
    if(meRegion[os.str()]) meRegion[os.str()]->Fill( totalNumberOfRecHits[i]);
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


