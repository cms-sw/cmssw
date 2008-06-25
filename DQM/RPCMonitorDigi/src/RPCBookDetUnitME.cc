#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>

using namespace std;
using namespace edm;
/// Booking of MonitoringElemnt for one RPCDetId (= roll)
map<string, MonitorElement*> RPCMonitorDigi::bookDetUnitME(RPCDetId & detId, const EventSetup & iSetup) {
  map<string, MonitorElement*> meMap;  

  string ringType = (detId.region() ==  0)?"Wheel":"Disk";

  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
  string folder = "RPC/RecHits/" +  folderStr->folderStructure(detId);

  dbe->setCurrentFolder(folder);
  
  //get number of strips in current roll
  int nstrips = this->stripsInRoll(detId, iSetup);
  if (nstrips == 0 ) nstrips = 1;

  /// Name components common to current RPCDetId  
   RPCGeomServ RPCname(detId);
  string nameRoll = RPCname.name();

  RPCGeomServ RPCindex(detId);
  int index = RPCindex.chambernr();

  stringstream os;
  if (dqmexpert) {    
    os.str("");
    os<<"Occupancy_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), nstrips, 0.5, nstrips+0.5);
    
    os.str("");
    os<<"BXN_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 21, -10.5, 10.5);

    os.str("");
    os<<"ClusterSize_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 20, 0.5, 20.5);
  
    os.str("");
    os<<"NumberOfClusters_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 10, 0.5, 10.5);

    os.str("");
    os<<"NumberOfDigi_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 10, 0.5, 10.5);
        
    os.str("");
    os<<"CrossTalkLow_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(),nstrips, 0.5,nstrips+0.5 );
    
    os.str("");
    os<<"CrossTalkHigh_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), nstrips, 0.5, nstrips+0.5);
    
    os.str("");
    os<<"BXWithData_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 11, -5.5, 5.5);

    /// RPCRecHits
    os.str("");
    os<<"RecHitXPosition_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 80, -120, 120);
    
    os.str("");
    os<<"RecHitDX_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 30, -10, 10);
    
    os.str("");
    os<<"RecHitCounter_"<<nameRoll;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(),nstrips,-0.5,nstrips+0.5);
  }
  
  if (dqmsuperexpert) {    
    os.str("");
    os<<"BXN_vs_strip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),  nstrips , 0.5, nstrips+0.5 , 21, -10.5, 10.5);
        
    os.str("");
    os<<"ClusterSize_vs_LowerSrip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),  nstrips, 0.5,  nstrips+0.5,11, 0.5, 11.5);
    
    os.str("");
    os<<"ClusterSize_vs_HigherStrip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), nstrips, 0.5,  nstrips+0.5,11, 0.5, 11.5);
    
    os.str("");
    os<<"ClusterSize_vs_Strip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),nstrips, 0.5, nstrips+0.5,11, 0.5, 11.5);
    
    os.str("");
    os<<"ClusterSize_vs_CentralStrip_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), nstrips, 0.5, nstrips+0.5,11, 0.5, 11.5);
 
    /// RPCRecHits
    os.str("");
    os<<"MissingHits_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),nstrips , 0, nstrips, 2, 0.,2.);
    
    os.str("");
    os<<"RecHitX_vs_dx_"<<nameRoll;
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),30, -100, 100,30,10,10);
  }
  
  dbe->setCurrentFolder(GlobalHistogramsFolder);
  
  os.str("");
  os<<"SectorOccupancy_"<<ringType<<"_"<<detId.ring()<<"_Sector_"<<detId.sector(); 

  //check if ME already exists for this sector
  MonitorElement * me = dbe->get(folder+"/"+os.str());
  if(me) return meMap;

  if (detId.sector()==9 || detId.sector()==11 )
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5,96.5, 15, 0.5, 15.5);
  else  if (detId.sector()==4) 
    meMap[os.str()] = dbe->book2D(os.str(), os.str(),  96, 0.5, 96.5, 21, 0.5, 21.5);
  else
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5,  96.5, 17, 0.5, 17.5);
    
  return meMap;
}

map<string, MonitorElement*> RPCMonitorDigi::bookRegionRing(int region, int ring) {  
  map<string, MonitorElement*> meMap;  
  string ringType = (region ==  0)?"Wheel":"Disk";
  
  dbe->setCurrentFolder(GlobalHistogramsFolder);
  
  stringstream os;
  
  os<<"WheelOccupancyXY_"<<ringType<<"_"<<region<<"_"<<ring;
  meMap[os.str()] = dbe->book2D(os.str(), os.str(),63, -800, 800, 63, -800, 800);

  os.str("");
  os<<"WheelClusterSize_"<<ringType<<"_"<<region<<"_"<<ring;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(),20, 0.5, 20.5);

  os.str("");
  os<<"Wheel1DOccupancy_"<<ringType<<"_"<<region<<"_"<<ring;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 12, 0.5, 12.5);

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
