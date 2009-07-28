#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <DQM/RPCMonitorDigi/interface/utils.h>
using namespace std;
using namespace edm;
/// Booking of MonitoringElemnt for one RPCDetId (= roll)
map<string, MonitorElement*> RPCMonitorDigi::bookDetUnitME(RPCDetId & detId, const EventSetup & iSetup) {
  map<string, MonitorElement*> meMap;  

  string ringType;
  int ring;
  if(detId.region() == 0) {
      ringType = "Wheel";  
    ring = detId.ring();
  }else if (detId.region() == -1){  
    ringType =  "Disk";
    ring = detId.region()*detId.station();
  }else {
    ringType =  "Disk";
    ring = detId.station();
  }

  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
  string folder = "RPC/RecHits/" +  folderStr->folderStructure(detId);

  dbe->setCurrentFolder(folder);
  
  //get number of strips in current roll
  int nstrips = this->stripsInRoll(detId, iSetup);
  if (nstrips == 0 ) nstrips = 1;

  /// Name components common to current RPCDetId  
  RPCGeomServ RPCname(detId);
  string nameRoll = RPCname.name();

  
  stringstream os;
  os.str("");
  os<<"Occupancy_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), nstrips, 0.5, nstrips+0.5);
  dbe->tag( meMap[os.str()],  rpcdqm::OCCUPANCY);

  //cout<<meMap[os.str()]->flags()<<endl;


  if (dqmexpert) {    
    os.str("");
    os<<"BXN_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 21, -10.5, 10.5);

    os.str("");
    os<<"ClusterSize_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 20, 0.5, 20.5);
   dbe->tag( meMap[os.str()],  rpcdqm::CLUSTERSIZE);
    os.str("");
    os<<"NumberOfClusters_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 10, 0.5, 10.5);

    os.str("");
    os<<"NumberOfDigi_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 50, 0.5, 50.5);
    dbe->tag( meMap[os.str()],  rpcdqm::MULTIPLICITY);

    
    os.str("");
    os<<"BXWithData_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 10, 0.5, 10.5);

    /// RPCRecHits

    os.str("");
    os<<"RecHitCounter_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(),20,0.5,20.5);
  }
  
  if (dqmsuperexpert) {    
    os.str("");
    os<<"BXN_vs_strip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),  nstrips , 0.5, nstrips+0.5 , 21, -10.5, 10.5);
     
    os.str("");
    os<<"ClusterSize_vs_Strip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),nstrips, 0.5, nstrips+0.5,11, 0.5, 11.5);
    
  }

  MonitorElement * myMe;

  os.str("");
  if(detId.region()==0)
    os<<"RPC/RecHits/Barrel/Wheel_"<<ring<<"/SummaryBySectors/";
  else if (detId.region()==1)
    os<<"RPC/RecHits/Endcap+/Disk_"<<ring<<"/SummaryBySectors/";
  else 
    os<<"RPC/RecHits/Endcap-/Disk_"<<ring<<"/SummaryBySectors/";
  string WheelSummary = os.str();
  dbe->setCurrentFolder(WheelSummary);
  
  os.str("");
  os<<"Occupancy_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
  myMe = dbe->get(WheelSummary+"/"+os.str());
  
  //check if ME for this sector have already been booked
  if(myMe)  meMap[os.str()]=myMe;
  else {
    if(detId.region()==0){
      if (detId.sector()==9 || detId.sector()==11)
	meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5,96.5, 15, 0.5, 15.5);
      else  if (detId.sector()==4) 
	meMap[os.str()] = dbe->book2D(os.str(), os.str(),  96, 0.5, 96.5, 21, 0.5, 21.5);
      else
	meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5,  96.5, 17, 0.5, 17.5);
    }else{//Endcap
	meMap[os.str()] = dbe->book2D(os.str(), os.str(), 32, 0.5, 32.5, 54, 0.5, 54.5);
    }
  }
  
  os.str("");
  os<<"BxDistribution_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
  myMe = dbe->get(folder+"/"+os.str());
  if(myMe)  meMap[os.str()]=myMe;
  else meMap[os.str()] = dbe->book1D(os.str(), os.str(), 11, -5.5, 5.5);

  os.str("");
  os<<"BXWithData_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
  myMe = dbe->get(folder+"/"+os.str());
  if(myMe)  meMap[os.str()]=myMe;
  else  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 10, 0.5, 10.5);

  return meMap;
}




map<string, MonitorElement*> RPCMonitorDigi::bookRegionRing(int region, int ring) {  
  map<string, MonitorElement*> meMap;  
  string ringType = (region ==  0)?"Wheel":"Disk";

  dbe->setCurrentFolder(GlobalHistogramsFolder);
  stringstream os, label;

  rpcdqm::utils mylabel;
  mylabel.dolabeling();

  // os<<"OccupancyXY_"<<ringType<<"_"<<ring;
//   //  meMap[os.str()] = dbe->book2D(os.str(), os.str(),63, -800, 800, 63, -800, 800);
//     meMap[os.str()] = dbe->book2D(os.str(), os.str(),1000, -800, 800, 1000, -800, 800);

  os.str("");
  os<<"ClusterSize_"<<ringType<<"_"<<ring;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(),20, 0.5, 20.5);

  os.str("");
  os<<"1DOccupancy_"<<ringType<<"_"<<ring;
  if (region!=0)  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 6, 0.5, 6.5);
  else meMap[os.str()] = dbe->book1D(os.str(), os.str(), 12, 0.5, 12.5);
  
  if(region==0) {
    
    os.str("");
    os<<"Occupancy_Roll_vs_Sector_"<<ringType<<"_"<<ring;                                      // new Occupancy Roll vs Sector
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), 12, 0.5,12.5, 21, 0.5, 21.5);
    for(int i=1; i<22; i++) {
      meMap[os.str()] ->setBinLabel(i, mylabel.YLabel(i), 2);
      if(i<13) {
	label.str("");
	label<<"Sec"<<i;
	meMap[os.str()] ->setBinLabel(i, label.str(), 1);
      }
    }
  } //end of Barrel 
    
    os.str("");
    os<<"BxDistribution_"<<ringType<<"_"<<ring;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 11, -5.5, 5.5);
    
  
  return meMap; 
}

//returns the number of strips in each roll
int  RPCMonitorDigi::stripsInRoll(RPCDetId & id, const EventSetup & iSetup) {
  edm::ESHandle<RPCGeometry> rpcgeo;
  iSetup.get<MuonGeometryRecord>().get(rpcgeo);

  const RPCRoll * rpcRoll = rpcgeo->roll(id);

  if (rpcRoll)
    return  rpcRoll->nstrips();
  else 
    return 1;
}
