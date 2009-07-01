 /***********************************************
 *						*
 *  implementation of RPCMonitorDigi class	*
 *						*
 ***********************************************/
#include <TRandom.h>
#include <string>
#include <sstream>
#include <set>
#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"
///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
///Geometry
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
///Log messages
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace edm;
RPCMonitorDigi::RPCMonitorDigi( const ParameterSet& pset ):counter(0){
  foundHitsInChamber.clear();
  nameInLog = pset.getUntrackedParameter<string>("moduleLogName", "RPC_DQM");

  saveRootFile  = pset.getUntrackedParameter<bool>("DigiDQMSaveRootFile", false); 
  mergeRuns_  = pset.getUntrackedParameter<bool>("MergeDifferentRuns", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("DigiEventsInterval", 10000);
  RootFileName  = pset.getUntrackedParameter<string>("RootFileNameDigi", "RPCMonitor.root"); 

  dqmshifter = pset.getUntrackedParameter<bool>("dqmshifter", false);
  dqmexpert = pset.getUntrackedParameter<bool>("dqmexpert", false);
  dqmsuperexpert = pset.getUntrackedParameter<bool>("dqmsuperexpert", false);

  RPCDataLabel = pset.getUntrackedParameter<std::string>("RecHitLabel","rpcRecHitLabel");
  digiLabel = pset.getUntrackedParameter<std::string>("DigiLabel","muonRPCDigis");

}

RPCMonitorDigi::~RPCMonitorDigi(){}

void RPCMonitorDigi::beginJob(EventSetup const&){
  LogInfo (nameInLog) <<"[RPCMonitorDigi]: Begin job" ;
  
  /// get hold of back-end interface
  dbe = Service<DQMStore>().operator->();

  GlobalHistogramsFolder="RPC/RecHits/SummaryHistograms";
  dbe->setCurrentFolder(GlobalHistogramsFolder);  

  ClusterSize_for_Barrel = dbe->book1D("ClusterSize_for_Barrel", "ClusterSize for Barrel", 20, 0.5, 20.5);
  ClusterSize_for_EndcapPositive = dbe->book1D("ClusterSize_for_EndcapPositive", "ClusterSize for PositiveEndcap",  20, 0.5, 20.5);
  ClusterSize_for_EndcapNegative = dbe->book1D("ClusterSize_for_EndcapNegative", "ClusterSize for NegativeEndcap", 20, 0.5, 20.5);

  ClusterSize_for_BarrelandEndcaps = dbe->book1D("ClusterSize_for_BarrelandEndcap", "ClusterSize for Barrel&Endcaps", 20, 0.5, 20.5);

  NumberOfClusters_for_Barrel = dbe -> book1D("NumberOfClusters_for_Barrel", "NumberOfClusters for Barrel", 20, 0.5, 20.5);
  NumberOfClusters_for_EndcapPositive = dbe -> book1D("NumberOfClusters_for_EndcapPositive", "NumberOfClusters for Endcap Positive", 20, 0.5, 20.5);
  NumberOfClusters_for_EndcapNegative = dbe -> book1D("NumberOfClusters_for_EndcapNegative", "NumberOfClusters for Endcap Negative", 20, 0.5, 20.5);

  BarrelNumberOfDigis = dbe -> book1D("Barrel_NumberOfDigi", "Number Of Digis in Barrel", 50, 0.5, 50.5);
  
  SameBxDigisMeBarrel_ = dbe->book1D("SameBXDigis_Barrel", "Digis with same bx", 20, 0.5, 20.5);  
 //  SameBxDigisMeEndcapPositive_ = dbe->book1D("SameBXDigis_EndcapPositive", "Digis with same bx", 20, 0.5, 20.5);  
 //   SameBxDigisMeEndcapNegative_ = dbe->book1D("SameBXDigis_EndcapNegative", "Digis with same bx", 20, 0.5, 20.5);  

  BarrelOccupancy = dbe -> book2D("Occupancy_for_Barrel", "Barrel Occupancy Wheel vs Sector", 12, 0.5, 12.5, 5, -2.5, 2.5);
  EndcapPositiveOccupancy = dbe -> book2D("Occupancy_for_EndcapPositive", "Endcap Positive Occupancy Disk vs Sector", 6, 0.5, 6.5, 4, 0.5, 4.5);
  EndcapNegativeOccupancy = dbe -> book2D("Occupancy_for_EndcapNegative", "Endcap Negative Occupancy Disk vs Sector", 6, 0.5, 6.5, 4, 0.5, 4.5);
  
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
}

void RPCMonitorDigi::beginRun(const Run& r, const EventSetup& iSetup){
  LogInfo (nameInLog) <<"Begin Run " ;
  //if mergeRuns_ skip reset
  //if merge remember to safe at job end and not at run end
  if (mergeRuns_) return;

  //MEs are reset at every new run. They are saved at the end of each run
  //Reset all Histograms
  for (map<uint32_t, map<string,MonitorElement*> >::const_iterator meItr = meCollection.begin();
       meItr!= meCollection.end(); meItr++){
    for (map<string,MonitorElement*>::const_iterator Itr = (*meItr).second.begin();
	 Itr!= (*meItr).second.end(); Itr++){
      (*Itr).second->Reset();
    }
  }

  for (map<pair<int,int>, map<string, MonitorElement*> >::const_iterator meItr =  meWheelDisk.begin();
       meItr!=  meWheelDisk.end(); meItr++){
    for (map<string,MonitorElement*>::const_iterator Itr = (*meItr).second.begin();
	 Itr!= (*meItr).second.end(); Itr++){
      (*Itr).second->Reset();
    }
  }
  
  //Reset All Global histos !!!!!!!!!!!!!!!!!!!!!!
  ClusterSize_for_Barrel->Reset();
  ClusterSize_for_EndcapPositive ->Reset();
  ClusterSize_for_EndcapNegative->Reset();
  ClusterSize_for_BarrelandEndcaps->Reset(); 

  NumberOfClusters_for_Barrel ->Reset();
  NumberOfClusters_for_EndcapPositive->Reset(); 
  NumberOfClusters_for_EndcapNegative ->Reset();


  SameBxDigisMeBarrel_->Reset();
  //  SameBxDigisMeEndcapPositive_->Reset();
  //   SameBxDigisMeEndcapNegative_ ->Reset();
  
  BarrelOccupancy ->Reset();
  EndcapPositiveOccupancy ->Reset();
  EndcapNegativeOccupancy->Reset();

  ESHandle<RPCGeometry> rpcGeo;
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
	
	if(rpcId.region() == 0) {
	  ring = rpcId.ring();
	}else if (rpcId.region() == -1){
	  ring = rpcId.region()*rpcId.station();
	}else{
	  ring = rpcId.station();
	}
	
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

void RPCMonitorDigi::analyze(const Event& iEvent,const EventSetup& iSetup ){

  counter++;
  LogInfo (nameInLog) <<"[RPCMonitorDigi]: Beginning analyzing event " << counter;
  
  RPCEvents -> Fill(1);
  
  /// RPC Geometry
  ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  /// Digis
  Handle<RPCDigiCollection> rpcdigis;
  iEvent.getByType(rpcdigis);

  //RecHits
  Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByType(rpcHits);

  map<int,int> bxMap;
 
  //Loop on digi collection
  for( RPCDigiCollection::DigiRangeIterator collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){
  
    RPCDetId detId=(*collectionItr).first; 
    uint32_t id=detId(); 

    const GeomDet* gdet=rpcGeo->idToDet(detId);
    const BoundPlane & surface = gdet->surface();
    
    //get roll name
    RPCGeomServ RPCname(detId);
    string nameRoll = RPCname.name();
    string YLabel = RPCname.shortname();
    stringstream os;

    //get roll number
    rpcdqm::utils prova;
    int nr = prova.detId2RollNr(detId);

    prova.fillvect();
    vector<int> SectStr2 = prova.SectorStrips2();
    vector<int> SectStr1 = prova.SectorStrips1();
    
    //get MEs corresponding to present detId  
    map<string, MonitorElement*> meMap=meCollection[id]; 
    if(meMap.size()==0) continue; 

    int region=detId.region();
    int ring;
    string regionName;
    string ringType;
    if(region == 0) {
      regionName="Barrel";  
      ringType = "Wheel";  
      ring = detId.ring();
    }else if (region == -1){
      regionName="Encap-";
      ringType =  "Disk";
      ring = region*detId.station();
    }else{
      regionName="Encap+";
      ringType =  "Disk";
      ring = detId.station();
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
    int bins;
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
      os.str("");
      os<<"Sec"<<detId.sector();
      meRingMap[meId] ->setBinLabel(detId.sector(), os.str(), 1);
      }

      os.str("");
      os<<"BxDistribution_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
      if(meRingMap[os.str()])
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
      if(meMap[os.str()])
	meMap[os.str()]->Fill(strip, nr);
    

     
 
      if(meMap[os.str()])
	meMap[os.str()]->setBinLabel(nr,YLabel, 2);


      os.str("");
      os<<"Occupancy_"<<nameRoll;
      if(meMap[os.str()]) meMap[os.str()]->Fill(strip);
      bins = meMap[os.str()]->getNbinsX();
      
      os.str("");
      os<<"Occupancy_Roll_vs_Sector_"<<ringType<<"_"<<ring;       
      if ( meRingMap[os.str()]) {
	meRingMap[os.str()]->Fill(detId.sector(), nr, 1);
	//if(detId.region()==0) meRingMap[os.str()]->setBinLabel(nr,YLabel, 2);
      }
      
      // moved to client side. remember to delete here !!!!!!!
      //     // do Occupancy normalization
      //       int SectSt;
      //       if(ring==2 || ring ==-2) SectSt = SectStr2[detId.sector()];     //get # of strips for given Sector
      //       else  SectSt= SectStr1[detId.sector()];                         
      //       float NormOcc = ((meRingMap[os.str()]->getBinContent(detId.sector(), nr)) / bins)/counter; // normalization by Strips & RPC Events
      
      //  os.str("");
      //       os<<"OccupancyNormByGeoAndEvents_Roll_vs_Sector_"<<ringType<<"_"<<ring; // Wrong place! must be in a client module!!!
      //       if(meRingMap[os.str()]) {
      // 	meRingMap[os.str()]->setBinContent(detId.sector(), nr, NormOcc);
      // 	if(detId.region()==0) meRingMap[os.str()]->setBinLabel(nr,YLabel, 2);
      //       }

    
      if(dqmexpert){ 	
	os.str("");
	os<<"BXN_"<<nameRoll;
	if(meMap[os.str()])
	  meMap[os.str()]->Fill(bx);
	}
  
      if (dqmsuperexpert) {	
	os.str("");
	os<<"BXN_vs_strip_"<<nameRoll;
	if(meMap[os.str()])
	  meMap[os.str()]->Fill(strip,bx);
      }
    }  //end loop of digis of given roll
  
    if (dqmexpert){
     //  for(unsigned int stripIter=0;stripIter<duplicatedDigi.size(); ++stripIter){
// 	if( stripIter<(duplicatedDigi.size()-1) && duplicatedDigi[stripIter+1].first==duplicatedDigi[stripIter].first+1) {
// 	  os.str("");
// 	  os<<"CrossTalkHigh_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(duplicatedDigi[stripIter].first);	
// 	}
// 	if(stripIter>0 && duplicatedDigi[stripIter-1].first == duplicatedDigi[stripIter].first+1) {
// 	  os.str("");
// 	  os<<"CrossTalkLow_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(duplicatedDigi[stripIter].first);	
// 	}
//       }
      os.str("");
      os<<"BXWithData_"<<nameRoll;
      if(meMap[os.str()])
	meMap[os.str()]->Fill(bxs.size());
    }
 
    os.str("");
    os<<"BXWithData_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
    if(meMap[os.str()])
      meMap[os.str()]->Fill(bxs.size());
 

    if(numberOfDigi>50) numberOfDigi=50; //overflow
    // os.str("");
//     os<<"NumberOfDigi_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
//     if(meMap[os.str()])
//       meMap[os.str()]->Fill(numberOfDigi);
    
    os.str("");
    os<<"NumberOfDigi_"<<nameRoll;
    if(meMap[os.str()]) { 
      meMap[os.str()]->Fill(numberOfDigi);
      // float thierdstrips = bins/3;
//       
//       if(numberOfDigi > thierdstrips) {  // Multiplicity greater than 1/3 of Strips
// 	os.str("");
// 	os<<"NumberOfDigiGreaterThanThierdStrips_"<<ringType<<"_"<<ring;
// 	if( meRingMap[os.str()]) meRingMap[os.str()]->Fill(detId.sector(), nr, 1);
// 	
//       }
    }
    
    BarrelNumberOfDigis -> Fill(numberOfDigi);

    
    

 
    // Fill RecHit MEs   
    if(recHitCollection.first==recHitCollection.second ){   
 
      if(dqmsuperexpert) {
// 	os.str("");
// 	os<<"MissingHits_"<<nameRoll;
// 	if(meMap[os.str()])
// 	  meMap[os.str()]->Fill((int)(counter), 1.0);
      }
    }else{     
      //      foundHitsInChamber[id]=true;
 
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
 
// 	os.str("");
// 	os<<"OccupancyXY_"<<ringType<<"_"<<ring;
// 	if(meRingMap[os.str()])
// 	  meRingMap[os.str()]->Fill(globalHitPoint.x(),globalHitPoint.y());

	int mult=it->clusterSize();		  //cluster size plot => should be within 1-3	
	int firstStrip=it->firstClusterStrip();    //plot first Strip => should be flat
	//	float xposition=point.x();

	ClusterSize_for_BarrelandEndcaps -> Fill(mult);
	
// 	if(mult>5) {
// 	  os.str("");
// 	  os<<"ClusterSizeGreaterThan5_"<<ringType<<"_"<<ring; 
// 	  if( meRingMap[os.str()]) meRingMap[os.str()]->Fill(detId.sector(), nr, 1);
// 	  //if(detId.region()==0)  meRingMap[os.str()]->setBinLabel(nr, YLabel, 2);
// 	}

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

// 	os.str("");
// 	os<<"ClusterSize_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
// 	if(meMap[os.str()])
// 	  meMap[os.str()] -> Fill(mult);

	if (dqmsuperexpert) {
	  int centralStrip=firstStrip;
	  if(mult%2) {
	    centralStrip+= mult/2;
	  }else{	
	    float x = gRandom->Uniform(2);
	    centralStrip+=(x<1)? (mult/2)-1 : (mult/2);
	  }
	  // os.str("");
// 	  os<<"ClusterSize_vs_CentralStrip_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(centralStrip,mult);
	

	  os.str("");
	  os<<"ClusterSize_vs_Strip_"<<nameRoll;
	  if(meMap[os.str()])
	    for(int index=0; index<mult; ++index)
	      meMap[os.str()]->Fill(firstStrip+index,mult);

// 	  os.str("");
// 	  os<<"ClusterSize_vs_LowerSrip_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(firstStrip,mult);
	
// 	  os.str("");
// 	  os<<"ClusterSize_vs_HigherStrip_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(firstStrip+mult-1,mult);
	
	 //  os.str("");
// 	  os<<"RecHitX_vs_dx_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(xposition,error.xx());
	}

	if(dqmexpert) {
	  os.str("");
	  os<<"ClusterSize_"<<nameRoll;
	  if(meMap[os.str()])
	    meMap[os.str()]->Fill(mult);
	  
	 //  os.str("");
// 	  os<<"RecHitXPosition_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(xposition);
	  
// 	  os.str("");
// 	  os<<"RecHitDX_"<<nameRoll;
// 	  if(meMap[os.str()])
// 	    meMap[os.str()]->Fill(error.xx());	  
	  
	}
	numberOfHits++;
      }/// end loop on RPCRecHits for given roll
      

      if(dqmexpert) {	 
	os.str("");
	os<<"NumberOfClusters_"<<nameRoll;
	if(meMap[os.str()])
	  meMap[os.str()]->Fill(numbOfClusters);
	
	if(numberOfHits>5) numberOfHits=16;////////////!!!!!!!!!!!!!!!!!!!!!!!	
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
      
//       os.str("");
//       os<<"NumberOfClusters_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
//       if(meMap[os.str()])
//       meMap[os.str()]->Fill(numbOfClusters);
    }
  }/// end loop on RPC Digi Collection

  //adding new histo C.Carrillo & A. Cimmino
  for (map<int, int>::const_iterator myItr= bxMap.begin(); 
       myItr!=bxMap.end(); myItr++){
    SameBxDigisMeBarrel_ ->Fill((*myItr).second);///must be fixed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  } 
}
