 /************************************************
 *						*
 *  implementation of RPCMonitorDigi class	*
 *						*
 ************************************************/
#include <TRandom.h>
#include <string>
#include <sstream>
#include <set>
#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
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
}

RPCMonitorDigi::~RPCMonitorDigi(){}

void RPCMonitorDigi::beginJob(EventSetup const&){
  LogInfo (nameInLog) <<"Beginning DQMMonitorDigi " ;
  
  /// get hold of back-end interface
  dbe = Service<DQMStore>().operator->();

  GlobalHistogramsFolder="RPC/RecHits/SummaryHistograms";
  dbe->setCurrentFolder(GlobalHistogramsFolder);  

  ClusterSize_for_Barrel = dbe->book1D("ClusterSize_for_Barrel", "ClusterSize for Barrel", 20, 0.5, 20.5);
  ClusterSize_for_EndcapForward = dbe->book1D("ClusterSize_for_EndcapForward", "ClusterSize for ForwardEndcap",  20, 0.5, 20.5);
  ClusterSize_for_EndcapBackward = dbe->book1D("ClusterSize_for_EndcapBackward", "ClusterSize for BackwardEndcap", 20, 0.5, 20.5);
  ClusterSize_for_BarrelandEndcaps = dbe->book1D("ClusterSize_for_BarrelandEndcap", "ClusterSize for Barrel&Endcaps", 20, 0.5, 20.5);

  NumberOfDigis_for_Barrel = dbe -> book1D("NumberOfDigi_for_Barrel", "NumberOfDigis for Barrel", 20, 0.5, 20.5);
  NumberOfClusters_for_Barrel = dbe -> book1D("NumberOfClusters_for_Barrel", "NumberOfClusters for Barrel", 20, 0.5, 20.5);

  SameBxDigisMe_ = dbe->book1D("SameBXDigis", "Digis with same bx", 20, 0.5, 20.5);  

  BarrelOccupancy = dbe -> book2D("BarrelOccupancy", "Barrel Occupancy Wheel vs Sector", 12, 0.5, 12.5, 5, -2.5, 2.5);

  stringstream binLabel;
  for (int i = 1; i<13; i++){
    binLabel.str("");
    binLabel<<"Sec"<<i;
    BarrelOccupancy -> setBinLabel(i, binLabel.str(), 1);
    if(i<6){
      binLabel.str("");
      binLabel<<"Wheel"<<i;
      BarrelOccupancy -> setBinLabel(i+3, binLabel.str(), 2);
    }
  }
}


void RPCMonitorDigi::beginRun(const Run& r, const EventSetup& c){
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
  
  //Reset All Global histos
  ClusterSize_for_Barrel->Reset();
  ClusterSize_for_EndcapForward ->Reset();
  ClusterSize_for_EndcapBackward->Reset();
  ClusterSize_for_BarrelandEndcaps->Reset(); 
  SameBxDigisMe_->Reset();
}

void RPCMonitorDigi::endJob(void)
{
  if(saveRootFile) dbe->save(RootFileName); 
  dbe = 0;
}

void RPCMonitorDigi::analyze(const Event& iEvent,const EventSetup& iSetup ){
  counter++;
  LogInfo (nameInLog) <<"[RPCMonitorDigi]: Beginning analyzing event " << counter;
  
  map<uint32_t, bool >::iterator mapItrReset;
  for (mapItrReset = foundHitsInChamber.begin(); mapItrReset != foundHitsInChamber.end(); ++ mapItrReset) {
    mapItrReset->second=false;
  }
  
  /// RPC Geometry
  ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  /// DIGI     
  Handle<RPCDigiCollection> rpcdigis;
  iEvent.getByType(rpcdigis);

  /// RecHits
  Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByType(rpcHits);
  
  map<int,int> bxMap;
  int totalNumberDigi[3][1];

  for (int k=1; k<=3;k++){
    totalNumberDigi[k][1]=0;
  }

  RPCDigiCollection::DigiRangeIterator collectionItr;
  //Loop on digi collection
  for(collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){
    
    RPCDetId detId=(*collectionItr ).first; 
    uint32_t id=detId(); 
    
    const GeomDet* gdet=rpcGeo->idToDet(detId);
    const BoundPlane & surface = gdet->surface();
      
    //get roll name
    RPCGeomServ RPCname(detId);
    string nameRoll = RPCname.name();
    
    stringstream os;
    
    RPCGeomServ RPCnumber(detId);
    int nr = RPCnumber.chambernr();
    
    std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
    if (meItr == meCollection.end() || (meCollection.size()==0)) {
      meCollection[id]=bookDetUnitME(detId,iSetup );
    }
    std::map<std::string, MonitorElement*> meMap=meCollection[id];

    int region=detId.region();
    int ring;
    
    string regionName;
    string ringType;
    if(detId.region() == 0) {
      regionName="Barrel";  
      ringType = "Wheel";  
      ring = detId.ring();
    }else if (detId.region() == -1){
      regionName="Encap-";
      ringType =  "Disk";
      ring = detId.region()*detId.layer();
    }else{
      regionName="Encap+";
      ringType =  "Disk";
      ring = detId.layer();
    }
    
    std::pair<int,int> regionRing(region,ring);
    std::map<std::pair<int,int>, std::map<std::string,MonitorElement*> >::iterator meRingItr = meWheelDisk.find(regionRing);
    if (meRingItr == meWheelDisk.end() || (meWheelDisk.size()==0)) {
      meWheelDisk[regionRing]=bookRegionRing(region,ring);
    }

    map<std::string, MonitorElement*> meRingMap=meWheelDisk[regionRing];
    
    vector<int> strips;
    vector<int> bxs;
    strips.clear(); 
    bxs.clear();

    //get the RecHits associated to the roll
    typedef pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
    rangeRecHits recHitCollection =  rpcHits->get(detId);
  
    int numberOfDigi= 0;

    //remove duplicated digis
    RPCDigiCollection::const_iterator digiItr; 
    //loop on digis of given roll
     for (digiItr =(*collectionItr ).second.first;digiItr != (*collectionItr ).second.second; ++digiItr){
      int strip= (*digiItr).strip();

      vector<int>::iterator itrStrips = find( strips.begin(),strips.end(),strip);
      if(itrStrips!=strips.end() && strips.size()!=0) continue;
      
      ++numberOfDigi;
      strips.push_back(strip);
      
      //get bx number for this digi
      int bx=(*digiItr).bx();
      vector<int>::iterator existingBX = find(bxs.begin(),bxs.end(),bx);
      if(existingBX==bxs.end())bxs.push_back(bx);
     
      //adding new histo C.Carrillo & A. Cimmino
      map<int,int>::const_iterator bxItr = bxMap.find((*digiItr).bx());
      if (bxItr == bxMap.end()|| bxMap.size()==0 )bxMap[(*digiItr).bx()]=1;
      else bxMap[(*digiItr).bx()]++;
  
      //sector based histograms for dqm shifter
      os.str("");
      os<<"1DOccupancy_"<<ringType<<"_"<<ring;
      meRingMap[os.str()]->Fill(detId.sector());
  
      os.str("");
      os<<"BxDistribution_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
      meMap[os.str()]->Fill(bx);
    
      os.str("");
      os<<"BxDistribution_"<<ringType<<"_"<<ring;
      meRingMap[os.str()]->Fill(bx);
      
      BarrelOccupancy -> Fill(detId.sector(), ring);
     
      os.str("");
      os<<"Occupancy_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
      meMap[os.str()]->Fill(strip, nr);
      string Yaxis=nameRoll;
      Yaxis.erase (1,1);
      Yaxis.erase(0,3);
      Yaxis.replace(Yaxis.find("S"),4,"");
      Yaxis.erase(Yaxis.find("_")+2,8);
      meMap[os.str()]->setBinLabel(nr, Yaxis, 2);
    
      os.str("");
      os<<"Occupancy_"<<nameRoll;
      meMap[os.str()]->Fill(strip);
    
      if(dqmexpert){ 	
	os.str("");
	os<<"BXN_"<<nameRoll;
	meMap[os.str()]->Fill(bx);
	}

      if (dqmsuperexpert) {	
	os.str("");
	os<<"BXN_vs_strip_"<<nameRoll;
	meMap[os.str()]->Fill(strip,bx);
      }
    }  //end loop of digis of given roll
    
    if (dqmexpert){

      for(unsigned int stripIter=0;stripIter<strips.size(); ++stripIter){	
	if(stripIter< (strips.size()-1) && strips[stripIter+1]==strips[stripIter]+1) {
	  os.str("");
	  os<<"CrossTalkHigh_"<<nameRoll;
	  meMap[os.str()]->Fill(strips[stripIter]);	
	}
	if(stripIter >0 && strips[stripIter-1]==strips[stripIter]-1) {
	  os.str("");
	  os<<"CrossTalkLow_"<<nameRoll;
	      meMap[os.str()]->Fill(strips[stripIter]);	
	}
	  }
      
      os.str("");
      os<<"NumberOfDigi_"<<nameRoll;
      meMap[os.str()]->Fill(numberOfDigi);
      
      os.str("");
      os<<"BXWithData_"<<nameRoll;
      meMap[os.str()]->Fill(bxs.size());
    }
    
    os.str("");
    os<<"BXWithData_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
    meMap[os.str()]->Fill(bxs.size());

    LogInfo (nameInLog) <<"------------------" << numberOfDigi;

    os.str("");
    os<<"NumberOfDigi_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
    meMap[os.str()]->Fill(numberOfDigi);

    totalNumberDigi[detId.region()+2][1]+= numberOfDigi;	

    // Fill RecHit MEs   
    if(recHitCollection.first==recHitCollection.second ){   
      if(dqmsuperexpert) {
	os.str("");
	os<<"MissingHits_"<<nameRoll;
	meMap[os.str()]->Fill((int)(counter), 1.0);//////////!!!!!!!!!!!!!!!!!!!!!!!!!!!
      }
    }else{     
//  std::map<uint32_t, bool >::iterator mapItr = foundHitsInChamber.find(id);
//        if (mapItr == foundHitsInChamber.end() || (foundHitsInChamber.size()==0)) {/////////////////////////!!!!!!!!!!!!!!!!!!!!!!!
// 	 os.str("");
// 	 os<<"RecHitCounter_"<<nameRoll;
// 	 if(dqmexpert) meMap[os.str()]->setBinContent(1, counter);
//        }
      
      foundHitsInChamber[id]=true;
      
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
	
	os.str("");
	os<<"OccupancyXY_"<<ringType<<"_"<<ring;
	meRingMap[os.str()]->Fill(globalHitPoint.x(),globalHitPoint.y());
	
	int mult=it->clusterSize();		  //cluster size plot => should be within 1-3	
	int firstStrip=it->firstClusterStrip();    //plot first Strip => should be flat
	float xposition=point.x();
	
	ClusterSize_for_BarrelandEndcaps -> Fill(mult);
	
	if(detId.region() ==  0) {
	  ClusterSize_for_Barrel -> Fill(mult);
	} else if (detId.region() ==  -1) {
	  if(mult<=10) ClusterSize_for_EndcapBackward -> Fill(mult);
	  else ClusterSize_for_EndcapBackward -> Fill(11);	   
	} else if (detId.region() ==  1) {
	  if(mult<=10) ClusterSize_for_EndcapForward -> Fill(mult);
	  else ClusterSize_for_EndcapForward -> Fill(11);
	} 
	
	//Cluster Size by Wheels and sector
	os.str("");
	os<<"ClusterSize_"<<ringType<<"_"<<ring;
	meRingMap[os.str()] -> Fill(mult); 
	
	os.str("");
	os<<"ClusterSize_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
	meMap[os.str()] -> Fill(mult);
	
	if (dqmsuperexpert) {
	  int centralStrip=firstStrip;
	  if(mult%2) {
	    centralStrip+= mult/2;
	  }else{	
	    float x = gRandom->Uniform(2);
	    centralStrip+=(x<1)? (mult/2)-1 : (mult/2);
	  }
	  os.str("");
	  os<<"ClusterSize_vs_CentralStrip_"<<nameRoll;
	  meMap[os.str()]->Fill(centralStrip,mult);
	  
	  for(int index=0; index<mult; ++index){
	    os.str("");
	    os<<"ClusterSize_vs_Strip_"<<nameRoll;
	    meMap[os.str()]->Fill(firstStrip+index,mult);
	  }
	  
	  os.str("");
	  os<<"ClusterSize_vs_LowerSrip_"<<nameRoll;
	  meMap[os.str()]->Fill(firstStrip,mult);
	  
	  os.str("");
	  os<<"ClusterSize_vs_HigherStrip_"<<nameRoll;
	  meMap[os.str()]->Fill(firstStrip+mult-1,mult);
	  
	  os.str("");
	  os<<"RecHitX_vs_dx_"<<nameRoll;
	  meMap[os.str()]->Fill(xposition,error.xx());
	}
	
	if(dqmexpert) {
	  os.str("");
	  os<<"ClusterSize_"<<nameRoll;
	  meMap[os.str()]->Fill(mult);
	  
	  os.str("");
	  os<<"RecHitXPosition_"<<nameRoll;
	  meMap[os.str()]->Fill(xposition);
	  
	  os.str("");
	  os<<"RecHitDX_"<<nameRoll;
	  meMap[os.str()]->Fill(error.xx());	   
	}
	
	numberOfHits++;
	
      }/// end loop on RPCRecHits for given roll
      
      
      if(dqmexpert) {	 
	os.str("");
	os<<"NumberOfClusters_"<<nameRoll;
	meMap[os.str()]->Fill(numbOfClusters);
	
	if(numberOfHits>5) numberOfHits=16;////////////!!!!!!!!!!!!!!!!!!!!!!!11
	
	os.str("");
	os<<"RecHitCounter_"<<nameRoll;
	meMap[os.str()]->Fill(numberOfHits);
      }
      
      NumberOfClusters_for_Barrel -> Fill(numbOfClusters);
      
      os.str("");
      os<<"NumberOfClusters_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
      meMap[os.str()]->Fill(numbOfClusters);
    }
  }/// end loop on RPC Digi Collection
  
  //fill global histo
  NumberOfDigis_for_Barrel ->Fill(totalNumberDigi[2][1]);

  //adding new histo C.Carrillo & A. Cimmino
  for (map<int, int>::const_iterator myItr= bxMap.begin(); 
       myItr!=bxMap.end(); myItr++){
    SameBxDigisMe_ ->Fill((*myItr).second);
  }
}
