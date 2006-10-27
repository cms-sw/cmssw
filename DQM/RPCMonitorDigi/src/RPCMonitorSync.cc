/** \file
 *
 *  implementation of RPCMonitorSync class
 *
 *  $Date: 2006/10/20 13:42:20 $
 *  $Revision: 0.1 $
 *
 * \author Piotr Traczyk
 */

#include <DQM/RPCMonitorDigi/interface/RPCMonitorSync.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include <vector>
#include <ctime>

using namespace std;
using namespace edm;

MonitorElement *RPCMonitorSync::barrelOffsetHist( char *name, char *title ) {
  char htit[128];
  sprintf(htit,"Synchronization offset - %s",title);
  dbe->setCurrentFolder("Global");
  MonitorElement *hist = dbe->book2D(name,title,18,1,7,24,1,13);
  return hist;
} 

MonitorElement *RPCMonitorSync::endcapOffsetHist( char *name, char *title ) {
  char htit[128];
  sprintf(htit,"Synchronization offset - %s",title);
  dbe->setCurrentFolder("Global");
  MonitorElement *hist = dbe->book2D(name,title,12,1,4,36,1,7);
  return hist;
} 

MonitorElement *RPCMonitorSync::barrelWidthHist( char *name, char *title ) {
  char htit[128];
  sprintf(htit,"Synchronization width - %s",title);
  dbe->setCurrentFolder("Global");
  MonitorElement *hist = dbe->book2D(name,title,18,1,7,24,1,13);
  return hist;
} 

MonitorElement *RPCMonitorSync::endcapWidthHist( char *name, char *title ) {
  char htit[128];
  sprintf(htit,"Synchronization width - %s",title);
  dbe->setCurrentFolder("Global");
  MonitorElement *hist = dbe->book2D(name,title,12,1,4,36,1,7);
  return hist;
} 


RPCMonitorSync::RPCMonitorSync( const edm::ParameterSet& pset )
{
  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPC_DQM");

  saveRootFile  = pset.getUntrackedParameter<bool>("SyncDQMSaveRootFile", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("SyncEventsInterval", 10000); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameSync", "RPCMonitorSync.root"); 
  
  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();
}

RPCMonitorSync::~RPCMonitorSync(){
  cout << "Destruct RPCMonitorSync..." << endl;
}


std::map<std::string, MonitorElement*> RPCMonitorSync::bookDetUnitME(RPCDetId & detId) {

  std::map<std::string, MonitorElement*> meMap;
 
  std::string regionName;
  std::string ringType;
  if(detId.region() ==  0) {
    regionName="Barrel";
    ringType="Wheel";
  } else {
    ringType="Disk";
    if(detId.region() == -1) regionName="Encap-";
    if(detId.region() ==  1) regionName="Encap+";
  }
 
  char  folder[120];
  sprintf(folder,"%s/%s_%d/station_%d/sector_%d",regionName.c_str(),ringType.c_str(),
 				detId.ring(),detId.station(),detId.sector());
 
  dbe->setCurrentFolder(folder);

  /// Name components common to current RPDDetId  
  char detUnitLabel[128];
  char layerLabel[128];
  sprintf(detUnitLabel ,"%d",detId());
  sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());

  char meId [128];
  char meTitle [128];
  
  /// BEgin booking
  sprintf(meId,"Sync_%s",detUnitLabel);
  sprintf(meTitle,"BX_Sychronization_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 9, -4.5, 4.5);
  
  return meMap;
}

void RPCMonitorSync::endJob(void)
{
  cout << "====== RPCMonitorSync::endJob" << endl;
  cout << "  synchroMap.size(): "<<synchroMap.size()<<endl;

  MonitorElement *hOffsetMBp1 = barrelOffsetHist("hOffsetMBp1","Barrell Wheel +1");
  MonitorElement *hOffsetMBp2 = barrelOffsetHist("hOffsetMBp2","Barrell Wheel +2");
  MonitorElement *hOffsetMB0  = barrelOffsetHist("hOffsetMB0" ,"Barrell Wheel 0");
  MonitorElement *hOffsetMBm1 = barrelOffsetHist("hOffsetMBm1","Barrell Wheel -1");
  MonitorElement *hOffsetMBm2 = barrelOffsetHist("hOffsetMBm2","Barrell Wheel -2");
  MonitorElement *hOffsetMEp1 = endcapOffsetHist("hOffsetMEp1","Endcap Disk +1");
  MonitorElement *hOffsetMEp2 = endcapOffsetHist("hOffsetMEp2","Endcap Disk +2");
  MonitorElement *hOffsetMEp3 = endcapOffsetHist("hOffsetMEp3","Endcap Disk +3");
  MonitorElement *hOffsetMEm1 = endcapOffsetHist("hOffsetMEm1","Endcap Disk -1");
  MonitorElement *hOffsetMEm2 = endcapOffsetHist("hOffsetMEm2","Endcap Disk -2");
  MonitorElement *hOffsetMEm3 = endcapOffsetHist("hOffsetMEm3","Endcap Disk -3");

  MonitorElement *hWidthMBp1 = barrelWidthHist("hWidthMBp1","Barrell Wheel +1");
  MonitorElement *hWidthMBp2 = barrelWidthHist("hWidthMBp2","Barrell Wheel +2");
  MonitorElement *hWidthMB0  = barrelWidthHist("hWidthMB0" ,"Barrell Wheel 0");
  MonitorElement *hWidthMBm1 = barrelWidthHist("hWidthMBm1","Barrell Wheel -1");
  MonitorElement *hWidthMBm2 = barrelWidthHist("hWidthMBm2","Barrell Wheel -2");
  MonitorElement *hWidthMEp1 = endcapWidthHist("hWidthMEp1","Endcap Disk +1");
  MonitorElement *hWidthMEp2 = endcapWidthHist("hWidthMEp2","Endcap Disk +2");
  MonitorElement *hWidthMEp3 = endcapWidthHist("hWidthMEp3","Endcap Disk +3");
  MonitorElement *hWidthMEm1 = endcapWidthHist("hWidthMEm1","Endcap Disk -1");
  MonitorElement *hWidthMEm2 = endcapWidthHist("hWidthMEm2","Endcap Disk -2");
  MonitorElement *hWidthMEm3 = endcapWidthHist("hWidthMEm3","Endcap Disk -3");
 
  map <int,timing>::const_iterator ci;
  float offset,width;
  float xf=0,yf=0;

  for(ci=synchroMap.begin();ci!=synchroMap.end();ci++){
    int id = ci->first;
    RPCDetId detId(id);
    
    offset = ci->second.offset();
    width = ci->second.width();

    RPCDetId *tempDetId=new RPCDetId(ci->first); 
        
    cout<< "id: " << ci->first << "    " << *tempDetId << "    offset: " << offset << "    width: " << width << endl;	
    
    int station_map[8]={1,2,3,4,5,0,6,0};
    
    if( detId.region()==0 ) {
      xf=station_map[detId.station()*2+detId.layer()-3]+((float)(detId.roll()-0.5)/3.);
      yf=detId.sector() +((float)(detId.subsector()-0.5)/2.);
      if ((detId.sector()==4) && (detId.station()==4)) 
        yf=detId.sector() +((float)(detId.subsector()-0.5)/4.);
      if (detId.ring()==1) hOffsetMBp1->Fill(xf,yf,offset);
      if (detId.ring()==2) hOffsetMBp2->Fill(xf,yf,offset);
      if (detId.ring()==-1) hOffsetMBm1->Fill(xf,yf,offset);
      if (detId.ring()==-2) hOffsetMBm2->Fill(xf,yf,offset);
      if (detId.ring()==0) hOffsetMB0->Fill(xf,yf,offset);
      if (detId.ring()==1) hWidthMBp1->Fill(xf,yf,width);
      if (detId.ring()==2) hWidthMBp2->Fill(xf,yf,width);
      if (detId.ring()==-1) hWidthMBm1->Fill(xf,yf,width);
      if (detId.ring()==-2) hWidthMBm2->Fill(xf,yf,width);
      if (detId.ring()==0) hWidthMB0->Fill(xf,yf,width);
    } else {
      xf=detId.ring()  +((float)(detId.roll()-0.5)/4.);
      yf=detId.sector()+((float)(detId.subsector()-0.5)/6.);
      if (detId.region()==1) {
        if (detId.station()==1) hOffsetMEp1->Fill(xf,yf,offset);
        if (detId.station()==2) hOffsetMEp2->Fill(xf,yf,offset);
        if (detId.station()==3) hOffsetMEp3->Fill(xf,yf,offset);
        if (detId.station()==1) hWidthMEp1->Fill(xf,yf,width);
        if (detId.station()==2) hWidthMEp2->Fill(xf,yf,width);
        if (detId.station()==3) hWidthMEp3->Fill(xf,yf,width);
      }
      if (detId.region()==-1) {
        if (detId.station()==1) hOffsetMEm1->Fill(xf,yf,offset);
        if (detId.station()==2) hOffsetMEm2->Fill(xf,yf,offset);
        if (detId.station()==3) hOffsetMEm3->Fill(xf,yf,offset);
        if (detId.station()==1) hWidthMEm1->Fill(xf,yf,width);
        if (detId.station()==2) hWidthMEm2->Fill(xf,yf,width);
        if (detId.station()==3) hWidthMEm3->Fill(xf,yf,width);
      }
    }
//    cout << "xf= "<<xf<<"   yf= "<<yf<<endl;  
  }

  if(saveRootFile) 
    dbe->save(RootFileName);

}


void RPCMonitorSync::readRPCDAQStrips(const edm::Event& iEvent) {

  timing aTiming;
  aTiming.early = 0;
  aTiming.inTime = 0;
  aTiming.late = 0;

  char detUnitLabel[128];
  char layerLabel[128];
  char meId [128];


  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByType(rpcDigis);
  RPCDigiCollection::DigiRangeIterator rpcDigiCI;
  for(rpcDigiCI = rpcDigis->begin();rpcDigiCI!=rpcDigis->end();rpcDigiCI++){
    cout<<(*rpcDigiCI).first<<endl;
    RPCDetId detId=(*rpcDigiCI).first; 
    uint32_t id=detId(); 

    sprintf(detUnitLabel ,"%d",detId());
    sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());
 
    std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
    if (meItr == meCollection.end() || (meCollection.size()==0)) {
      meCollection[id]=bookDetUnitME(detId);
    }
    std::map<std::string, MonitorElement*> meMap=meCollection[id];

    if(synchroMap.find(id)==synchroMap.end()) synchroMap[id] = aTiming;
//    cout << "synchroMap.size(): "<<synchroMap.size()<<endl;
    const RPCDigiCollection::Range& range = (*rpcDigiCI).second;
    int aBX=2;    
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;++digiIt){
      if( digiIt->bx()<aBX) aBX= digiIt->bx();
    }
    if(aBX==-1) synchroMap[id].early++;
    if(aBX==0) synchroMap[id].inTime++;
    if(aBX==1) synchroMap[id].late++;
    sprintf(meId,"Sync_%s",detUnitLabel);
    meMap[meId]->Fill(aBX);
  }
  if((!(counter%saveRootFileEventsInterval))&&(saveRootFile) ) dbe->save(RootFileName);
}

void RPCMonitorSync::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

   std::cout << "====== RPCMonitorSync" << std::endl;
   cout << "--- Run: " << iEvent.id().run()
	<< " Event: " << iEvent.id().event() 
	<< " time: "<<iEvent.time().value();
   time_t aTime = iEvent.time().value();
   cout<<" "<<ctime(&aTime)<<endl;
   
   cout<<"RPC digis: "<<endl;
   readRPCDAQStrips(iEvent);

   return;
}
