 /***********************************************
 *  implementation of RPCMonitorDigi class	*
 ***********************************************/
#include <TRandom.h>
#include <string>
#include <sstream>
#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
///Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;
RPCMonitorDigi::RPCMonitorDigi( const ParameterSet& pset ):counter(0){

  nameInLog = pset.getUntrackedParameter<string>("moduleLogName", "rpcmonitordigi");

  saveRootFile  = pset.getUntrackedParameter<bool>("DigiDQMSaveRootFile", false); 
  mergeRuns_  = pset.getUntrackedParameter<bool>("MergeDifferentRuns", false); 
  // saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("DigiEventsInterval", 10000);
  RootFileName  = pset.getUntrackedParameter<string>("RootFileNameDigi", "RPCMonitor.root"); 

  globalFolder_ = pset.getUntrackedParameter<string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms");
  muonNoise_ = pset.getUntrackedParameter<string>("DataType", "Noise");

  dqmshifter = pset.getUntrackedParameter<bool>("dqmshifter", false);
  dqmexpert = pset.getUntrackedParameter<bool>("dqmexpert", false);
  dqmsuperexpert = pset.getUntrackedParameter<bool>("dqmsuperexpert", false);

  RPCRecHitLabel_ = pset.getParameter<InputTag>("RecHitLabel");
  RPCDigiLabel_ =pset.getParameter<InputTag>("DigiLabel");
}

RPCMonitorDigi::~RPCMonitorDigi(){}



void RPCMonitorDigi::beginJob(){
  LogInfo (nameInLog) <<"[RPCMonitorDigi]: Begin job" ;
  
  /// get hold of back-end interface
  dbe = Service<DQMStore>().operator->();

  dbe->setCurrentFolder(globalFolder_);  

  ClusterSize_for_Barrel = dbe->book1D("ClusterSize_for_Barrel", "ClusterSize for Barrel", 20, 0.5, 20.5);
  ClusterSize_for_EndcapPositive = dbe->book1D("ClusterSize_for_EndcapPositive", "ClusterSize for PositiveEndcap",  20, 0.5, 20.5);
  ClusterSize_for_EndcapNegative = dbe->book1D("ClusterSize_for_EndcapNegative", "ClusterSize for NegativeEndcap", 20, 0.5, 20.5);

  ClusterSize_for_BarrelandEndcaps = dbe->book1D("ClusterSize_for_BarrelandEndcap", "ClusterSize for Barrel&Endcaps", 20, 0.5, 20.5);

  NumberOfClusters_for_Barrel = dbe -> book1D("NumberOfClusters_for_Barrel", "NumberOfClusters for Barrel", 20, 0.5, 20.5);
  NumberOfClusters_for_EndcapPositive = dbe -> book1D("NumberOfClusters_for_EndcapPositive", "NumberOfClusters for Endcap Positive", 20, 0.5, 20.5);
  NumberOfClusters_for_EndcapNegative = dbe -> book1D("NumberOfClusters_for_EndcapNegative", "NumberOfClusters for Endcap Negative", 20, 0.5, 20.5);
  
  NumberOfDigis_for_Barrel = dbe -> book1D("NumberOfDigi_for_Barrel", "Number Of Digis in Barrel", 50, 0.5, 50.5);
  NumberOfDigis_for_EndcapPositive = dbe -> book1D("NumberOfDigi_for_EndcapPositive", "Number Of Digis in EndCapPositive", 50, 0.5, 50.5);
  NumberOfDigis_for_EndcapNegative= dbe -> book1D("NumberOfDigi_for_EndcapNegative", "Number Of Digis in EndCapNegative", 50, 0.5, 50.5);
  
  SameBxDigisMeBarrel_ = dbe->book1D("SameBXDigis_Barrel", "Digis with same bx", 20, 0.5, 20.5);  
  SameBxDigisMeEndcapPositive_ = dbe->book1D("SameBXDigis_EndcapPositive", "Digis with same bx", 20, 0.5, 20.5);  
  SameBxDigisMeEndcapNegative_ = dbe->book1D("SameBXDigis_EndcapNegative", "Digis with same bx", 20, 0.5, 20.5);  

  BarrelOccupancy = dbe -> book2D("Occupancy_for_Barrel", "Barrel Occupancy Wheel vs Sector", 12, 0.5, 12.5, 5, -2.5, 2.5);
  EndcapPositiveOccupancy = dbe -> book2D("Occupancy_for_EndcapPositive", "Endcap Positive Occupancy Disk vs Sector", 6, 0.5, 6.5, 4, 0.5, 4.5);
  EndcapNegativeOccupancy = dbe -> book2D("Occupancy_for_EndcapNegative", "Endcap Negative Occupancy Disk vs Sector", 6, 0.5, 6.5, 4, 0.5, 4.5);

  dbe->setCurrentFolder(globalFolder_);   
  RPCEvents = dbe -> book1D("RPCEvents", "RPC Events Barrel+EndCap", 1, 0.5, 1.5);
 
  stringstream binLabel;
  for (int i = 1; i<13; i++){
    binLabel.str("");
    binLabel<<"Sec"<<i;
    BarrelOccupancy -> setBinLabel(i, binLabel.str(), 1);
    if(i<6){
      binLabel.str("");
      binLabel<<"Wheel"<<i-3;
      BarrelOccupancy -> setBinLabel(i, binLabel.str(), 2);
    }    
    if(i<7) {
      binLabel.str("");
      binLabel<<"Sec"<<i;
      EndcapPositiveOccupancy -> setBinLabel(i, binLabel.str(), 1);
      EndcapNegativeOccupancy -> setBinLabel(i, binLabel.str(), 1);
    }
      if(i<5){
      binLabel.str("");
      binLabel<<"Disk+"<<i ;                                 ;
      EndcapPositiveOccupancy -> setBinLabel(i, binLabel.str(), 2);
      binLabel.str("");
      binLabel<<"Disk-"<<i  ;  
      EndcapNegativeOccupancy -> setBinLabel(i, binLabel.str(), 2);
    }
  }
  dcs_ = true;

}

void RPCMonitorDigi::beginRun(const Run& r, const EventSetup& iSetup){

  LogInfo (nameInLog) <<"Begin Run " ;
  
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  //loop on geometry to book all MEs
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
	//std::cout<<"Booking for "<<nameRoll<<std::endl;
	meCollection[(uint32_t)rpcId]=bookDetUnitME(rpcId,iSetup );
 
	int ring;
	if(rpcId.region() == 0) 
	  ring = rpcId.ring();
	else 
	  ring = rpcId.region()*rpcId.station();
	
	//book wheel/disk histos
	std::pair<int,int> regionRing(region,ring);
	std::map<std::pair<int,int>, std::map<std::string,MonitorElement*> >::iterator meRingItr = meWheelDisk.find(regionRing);
	if (meRingItr == meWheelDisk.end() || (meWheelDisk.size()==0))  meWheelDisk[regionRing]=bookRegionRing(region,ring);
      }
    }
  }//end loop on geometry to book all MEs
}

void RPCMonitorDigi::endJob(void){
  if(saveRootFile) dbe->save(RootFileName); 
  dbe = 0;
}



void RPCMonitorDigi::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup ){
  
  this->makeDcsInfo(iEvent);
  if( !dcs_) return;//if RPC not ON there's no need to continue

  counter++;

  LogInfo (nameInLog) <<"[RPCMonitorDigi]: Beginning analyzing event " << counter;  

  /// Digis
  edm::Handle<RPCDigiCollection> rpcdigis;
  iEvent.getByLabel(RPCDigiLabel_, rpcdigis);

  //RecHits
  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(RPCRecHitLabel_,rpcHits);

  map<int,int> bxMap;
 
  if(rpcdigis->begin()!=rpcdigis->end())  RPCEvents -> Fill(1);

  //Loop on digi collection
  for( RPCDigiCollection::DigiRangeIterator collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){
  
    RPCDetId detId=(*collectionItr).first; 
    uint32_t id=detId(); 

    const GeomDet* gdet=rpcGeo->idToDet(detId);
    const BoundPlane & surface = gdet->surface();
    
    //get roll name
    RPCGeomServ RPCname(detId);
    string nameRoll = RPCname.name();
    //string YLabel = RPCname.shortname(); // to be removed later!!!
    stringstream os;

    //get roll number
    rpcdqm::utils prova;
    int nr = prova.detId2RollNr(detId);
    
    //get MEs corresponding to present detId  
    map<string, MonitorElement*> meMap=meCollection[id]; 
    if(meMap.size()==0) continue; 

    int region=detId.region();
    int ring;
    string ringType;
    if(region == 0) {
      ringType = "Wheel";  
      ring = detId.ring();
    }else{
      ringType =  "Disk";
      ring = region*detId.station();
    }
   
    //get wheel/disk MEs
    pair<int,int> regionRing(region,ring);
    map<string, MonitorElement*> meRingMap=meWheelDisk[regionRing];
    if(meRingMap.size()==0) continue;

    vector<pair <int,int> > duplicatedDigi;  
    vector<int> bxs;     

    //get the RecHits associated to the roll
    typedef pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
    rangeRecHits recHitCollection =  rpcHits->get(detId);
 
    int numberOfDigi= 0;

    RPCDigiCollection::const_iterator digiItr; 
    //loop on digis of given roll
    for (digiItr =(*collectionItr ).second.first;digiItr != (*collectionItr ).second.second; ++digiItr){
      int strip= (*digiItr).strip();
      int bx=(*digiItr).bx();
    
      //remove duplicated digis
      vector<pair <int,int> >::const_iterator itrDuplDigi = find(duplicatedDigi.begin(),duplicatedDigi.end(),make_pair(strip, bx));
      if(itrDuplDigi!=duplicatedDigi.end() && duplicatedDigi.size()!=0) continue;
    
      duplicatedDigi.push_back(make_pair(strip, bx));
      ++numberOfDigi;
  
      //bunch crossing
      vector<int>::const_iterator existingBX = find(bxs.begin(),bxs.end(),bx);
      if(existingBX==bxs.end())bxs.push_back(bx);
   
      //adding new histo C.Carrillo & A. Cimmino
      map<int,int>::const_iterator bxItr = bxMap.find((*digiItr).bx());
      if (bxItr == bxMap.end()|| bxMap.size()==0 )bxMap[(*digiItr).bx()]=1;
      else bxMap[(*digiItr).bx()]++;
   
      //sector based histograms for dqm shifter
      os.str("");
      os<<"1DOccupancy_"<<ringType<<"_"<<ring;
      string meId = os.str();
      if( meRingMap[meId]){
	meRingMap[meId]->Fill(detId.sector());
	// label
      }

      os.str("");
      os<<"BxDistribution_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
      if(meMap[os.str()])
	meMap[os.str()]->Fill(bx);
   
      os.str("");
      os<<"BxDistribution_"<<ringType<<"_"<<ring;
      if(meRingMap[os.str()])
	meRingMap[os.str()]->Fill(bx);
   
      if(detId.region()==0)
	BarrelOccupancy -> Fill(detId.sector(), ring);
      else if(detId.region()==1)
   	EndcapPositiveOccupancy -> Fill(detId.sector(), ring);
      else if(detId.region()==-1)
   	EndcapNegativeOccupancy -> Fill(detId.sector(),( -1 * ring) );//for RE- ring is negative 

      os.str("");
      os<<"Occupancy_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
      if(meMap[os.str()]){ 
	if(detId.region() ==0)	meMap[os.str()]->Fill(strip, nr);
	else	meMap[os.str()]->Fill(strip + 32*(detId.roll()-1),  RPCname.segment()+ ((detId.ring() -2)*6));
      }

      os.str("");
      os<<"Occupancy_"<<nameRoll;
      if(meMap[os.str()]) meMap[os.str()]->Fill(strip);
      
      os.str("");
      os<<"Occupancy_Roll_vs_Sector_"<<ringType<<"_"<<ring;       
      if (meRingMap[os.str()]) {
	meRingMap[os.str()]->Fill(detId.sector(), nr, 1);
      }
 
      os.str("");
      os<<"Occupancy_Ring_vs_Segment_"<<ringType<<"_"<<ring;   
       if (meRingMap[os.str()]) {
	meRingMap[os.str()]->Fill( RPCname.segment(), (detId.ring()-1)*3-detId.roll()+1, 1);
      }
    
      if(dqmexpert){ 	
	os.str("");
	os<<"BXN_"<<nameRoll;
	if(meMap[os.str()]) meMap[os.str()]->Fill(bx);
	}
  
      if (dqmsuperexpert) {	
	os.str("");
	os<<"BXN_vs_strip_"<<nameRoll;
	if(meMap[os.str()]) meMap[os.str()]->Fill(strip,bx);
      }
    }  //end loop of digis of given roll
  
    if (dqmexpert){
      os.str("");
      os<<"BXWithData_"<<nameRoll;
      if(meMap[os.str()]) meMap[os.str()]->Fill(bxs.size());
    }
 
    os.str("");
    os<<"BXWithData_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
    if(meMap[os.str()])
      meMap[os.str()]->Fill(bxs.size());

    if(numberOfDigi>50) numberOfDigi=50; //overflow
    
    os.str("");
    os<<"Multiplicity_"<<nameRoll;
    if(meMap[os.str()])   meMap[os.str()]->Fill(numberOfDigi);   
    
    if(detId.region()==0) NumberOfDigis_for_Barrel -> Fill(numberOfDigi);
    else  if(detId.region()==1) NumberOfDigis_for_EndcapPositive -> Fill(numberOfDigi);
    else  if(detId.region()==-1)  NumberOfDigis_for_EndcapNegative -> Fill(numberOfDigi);
                                
    // Fill RecHit MEs   
    if(recHitCollection.first!=recHitCollection.second ){   
 
      RPCRecHitCollection::const_iterator it;
      int numberOfHits=0;    
      int numbOfClusters=0;
      //loop RPCRecHits for given roll
      for (it = recHitCollection.first; it != recHitCollection.second ; it++) {
	numbOfClusters++; 
 	RPCDetId detIdRecHits=it->rpcId();
	LocalError error=it->localPositionError();//plot of errors/roll => should be gaussian	
	LocalPoint point=it->localPosition();     //plot of coordinates/roll =>should be flat
	GlobalPoint globalHitPoint=surface.toGlobal(point); 
 
	int mult=it->clusterSize();		  //cluster size plot => should be within 1-3	
	int firstStrip=it->firstClusterStrip();    //plot first Strip => should be flat

	ClusterSize_for_BarrelandEndcaps -> Fill(mult);
 
	if(detId.region() ==  0) {
	  ClusterSize_for_Barrel -> Fill(mult);
	} else if (detId.region() ==  -1) {
	  if(mult<=10) ClusterSize_for_EndcapNegative -> Fill(mult);
	  else ClusterSize_for_EndcapNegative -> Fill(11);	   
	} else if (detId.region() ==  1) {
	  if(mult<=10) ClusterSize_for_EndcapPositive -> Fill(mult);
	  else ClusterSize_for_EndcapPositive -> Fill(11);
	} 

	//Cluster Size by Wheels and sector
	os.str("");
	os<<"ClusterSize_"<<ringType<<"_"<<ring;
	if(meRingMap[os.str()])
	  meRingMap[os.str()] -> Fill(mult); 

	if (dqmsuperexpert) {
	  int centralStrip=firstStrip;
	  if(mult%2) {
	    centralStrip+= mult/2;
	  }else{	
	    float x = gRandom->Uniform(2);
	    centralStrip+=(x<1)? (mult/2)-1 : (mult/2);
	  }

	  os.str("");
	  os<<"ClusterSize_vs_Strip_"<<nameRoll;
	  if(meMap[os.str()])
	    for(int index=0; index<mult; ++index)
	      meMap[os.str()]->Fill(firstStrip+index,mult);
       	}

	if(dqmexpert) {
	  os.str("");
	  os<<"ClusterSize_"<<nameRoll;
	  if(meMap[os.str()])
	    meMap[os.str()]->Fill(mult);
	}
	numberOfHits++;
      }/// end loop on RPCRecHits for given roll
      
      if(dqmexpert) {	 
	if(numberOfHits>20) numberOfHits=20;
	os.str("");
	os<<"NumberOfClusters_"<<nameRoll;
	if(meMap[os.str()])
	  meMap[os.str()]->Fill(numbOfClusters);
	

	os.str("");
	os<<"RecHitCounter_"<<nameRoll;
	if(meMap[os.str()])
	  meMap[os.str()]->Fill(numberOfHits);
      }
      
      if(detId.region()==0)
	NumberOfClusters_for_Barrel -> Fill(numbOfClusters);
      else if (detId.region()==1)
	NumberOfClusters_for_EndcapPositive -> Fill(numbOfClusters);
      else if(detId.region()==-1)
	NumberOfClusters_for_EndcapNegative -> Fill(numbOfClusters);      

    }
  }/// end loop on RPC Digi Collection

  //adding new histo C.Carrillo & A. Cimmino
  for (map<int, int>::const_iterator myItr= bxMap.begin(); 
       myItr!=bxMap.end(); myItr++){
    SameBxDigisMeBarrel_ ->Fill((*myItr).second);///must be fixed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
                            dcsStatusItr != dcsStatus->end(); ++dcsStatusItr)   {

      if (!dcsStatusItr->ready(DcsStatus::RPC)) dcs_=false;
      
  }
      
  return ;
}
