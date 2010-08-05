#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <DQM/RPCMonitorDigi/interface/utils.h>

/// Booking of MonitoringElemnt for one RPCDetId (= roll)
std::map<std::string, MonitorElement*> RPCMonitorDigi::bookDetUnitME(RPCDetId & detId, const edm::EventSetup & iSetup, std::string recHitType) {
  std::map<std::string, MonitorElement*> meMap;  

  std::string ringType;
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
  std::string folder = subsystemFolder_+ "/"+folderStr->folderStructure(detId,recHitType);

  dbe->setCurrentFolder(folder);
  
  //get number of strips in current roll
  int nstrips = this->stripsInRoll(detId, iSetup);
  if (nstrips == 0 ) nstrips = 1;

  /// Name components common to current RPCDetId  
  RPCGeomServ RPCname(detId);
  std::string nameRoll = RPCname.name();
  
  std::stringstream os;
  os.str("");
  os<<"Occupancy_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), nstrips, 0.5, nstrips+0.5);
  dbe->tag( meMap[os.str()],  rpcdqm::OCCUPANCY);

  os.str("");
  os<<"BXDistribution_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 21, -10.5, 10.5);
  
  os.str("");
  os<<"ClusterSize_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 20, 0.5, 20.5);
  dbe->tag( meMap[os.str()],  rpcdqm::CLUSTERSIZE);
  
  os.str("");
  os<<"Multiplicity_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 50, 0.5, 50.5);
  dbe->tag( meMap[os.str()],  rpcdqm::MULTIPLICITY);
  
  os.str("");
  os<<"BXWithData_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 10, 0.5, 10.5);
  
  os.str("");
  os<<"NumberOfClusters_"<<nameRoll;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(),20,0.5,20.5);
  
  
  MonitorElement * myMe;

  os.str("");
  if(detId.region()==0)
    os<< subsystemFolder_<< "/"<<recHitType<<"/Barrel/Wheel_"<<ring<<"/SummaryBySectors/";
  else if (detId.region()==1)
    os<< subsystemFolder_<< "/"<<recHitType<<"/Endcap+/Disk_"<<ring<<"/SummaryBySectors/";
  else 
    os<< subsystemFolder_<< "/"<<recHitType<<"/Endcap-/Disk_"<<ring<<"/SummaryBySectors/";
  std::string WheelSummary = os.str();
  dbe->setCurrentFolder(WheelSummary);
  
  os.str("");
  os<<"Occupancy_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
  myMe = dbe->get(WheelSummary+"/"+os.str());
  
  rpcdqm::utils rpcUtils;
  //check if ME for this sector have already been booked
  if(myMe)  meMap[os.str()]=myMe;
  else {
    if(detId.region()==0) {
      if (detId.sector()==9 || detId.sector()==11)
	meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5,96.5, 15, 0.5, 15.5);
      else  if (detId.sector()==4) 
	meMap[os.str()] = dbe->book2D(os.str(), os.str(),  96, 0.5, 96.5, 21, 0.5, 21.5);
      else
	meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5,  96.5, 17, 0.5, 17.5);

      meMap[os.str()]->setAxisTitle("strip", 1);
      rpcUtils.labelYAxisRoll( meMap[os.str()], 0, ring);

    }else{//Endcap
      float fBin = ((detId.sector()-1)*6)+ 0.5;
      float lBin = fBin+12;
      meMap[os.str()] = dbe->book2D(os.str(), os.str(), 96, 0.5, 96.5, 12,fBin, lBin);
      meMap[os.str()]->setAxisTitle("strip", 1);
      std::stringstream yLabel;
      for(int r = 2; r<= 3; r ++) {
	int offset = 0;
	if (r ==3) offset =6;
	for (int i = 1 ; i<=6; i++) {
	  yLabel.str("");
	  yLabel<<"R"<<r<<"_C"<<(((detId.sector()-1)*6) +i);
	  meMap[os.str()]->setBinLabel(i+offset, yLabel.str(), 2);
	  
	}
      }
      for(int i = 1; i <= 96 ; i++) {
	if (i ==1) meMap[os.str()]->setBinLabel(i, "1", 1);
	else if (i==16) meMap[os.str()]->setBinLabel(i, "RollA", 1);
	else if (i==32) meMap[os.str()]->setBinLabel(i, "32", 1);
	else if (i==33) meMap[os.str()]->setBinLabel(i, "1", 1);
	else if (i==48) meMap[os.str()]->setBinLabel(i, "RollB", 1);
	else if (i==64) meMap[os.str()]->setBinLabel(i, "32", 1);
	else if (i==65) meMap[os.str()]->setBinLabel(i, "1", 1);
	else if (i==80) meMap[os.str()]->setBinLabel(i, "RollC", 1);
	else if (i==96) meMap[os.str()]->setBinLabel(i, "32", 1);
	else  meMap[os.str()]->setBinLabel(i, "", 1);
      }
    } 
  }
  
  os.str("");
  os<<"BxDistribution_"<<ringType<<"_"<<ring<<"_Sector_"<<detId.sector();
  myMe = dbe->get(WheelSummary+"/"+os.str());
  if(myMe)  meMap[os.str()]=myMe;
  else meMap[os.str()] = dbe->book1D(os.str(), os.str(), 11, -5.5, 5.5);

  return meMap;
}



std::map<std::string, MonitorElement*> RPCMonitorDigi::bookRegionRing(int region, int ring,std::string recHitType) {  
  std::map<std::string, MonitorElement*> meMap;  
  std::string ringType = (region ==  0)?"Wheel":"Disk";

  dbe->setCurrentFolder(subsystemFolder_ +"/"+recHitType+"/"+ globalFolder_);
  std::stringstream os, label;

  rpcdqm::utils rpcUtils;

  //     os<<"OccupancyXY_"<<ringType<<"_"<<ring;
  //     meMap[os.str()] = dbe->book2D(os.str(), os.str(),63, -800, 800, 63, -800, 800);
  //     meMap[os.str()] = dbe->book2D(os.str(), os.str(),1000, -800, 800, 1000, -800, 800);
  
  os.str("");
  os<<"ClusterSize_"<<ringType<<"_"<<ring;
  meMap[os.str()] = dbe->book1D(os.str(), os.str(),20, 0.5, 20.5);

  os.str("");
  os<<"1DOccupancy_"<<ringType<<"_"<<ring;
  if (region!=0)  meMap[os.str()] = dbe->book1D(os.str(), os.str(), 6, 0.5, 6.5);
  else meMap[os.str()] = dbe->book1D(os.str(), os.str(), 12, 0.5, 12.5);
  int sect=7;
  if(region==0) sect=13;
  for(int i=1; i<sect; i++) {
    label.str("");
    label<<"Sec"<<i;
    //cout<<label.str()<<endl;
    meMap[os.str()] ->setBinLabel(i, label.str(), 1); // to be corrected !!!!
  }
  
  if(region==0) {
    
    os.str("");
    os<<"Occupancy_Roll_vs_Sector_"<<ringType<<ring;                                   
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), 12, 0.5,12.5, 21, 0.5, 21.5);
    rpcUtils.labelXAxisSector(meMap[os.str()]);
    rpcUtils.labelYAxisRoll( meMap[os.str()], 0, ring);


  }else{
    
    os.str("");
    os<<"Occupancy_Ring_vs_Segment_"<<ringType<<"_"<<ring;                                  
    meMap[os.str()] = dbe->book2D(os.str(), os.str(), 36, 0.5,36.5, 6, 0.5, 6.5);
 
    rpcUtils.labelXAxisSegment(meMap[os.str()]);
    rpcUtils.labelYAxisRing(meMap[os.str()], 2);
  }
    
    os.str("");
    os<<"BxDistribution_"<<ringType<<"_"<<ring;
    meMap[os.str()] = dbe->book1D(os.str(), os.str(), 11, -5.5, 5.5);
    
  
  return meMap; 
}

//returns the number of strips in each roll
int  RPCMonitorDigi::stripsInRoll(RPCDetId & id, const edm::EventSetup & iSetup) {
  edm::ESHandle<RPCGeometry> rpcgeo;
  iSetup.get<MuonGeometryRecord>().get(rpcgeo);

  const RPCRoll * rpcRoll = rpcgeo->roll(id);

  if (rpcRoll)
    return  rpcRoll->nstrips();
  else 
    return 1;
}



void  RPCMonitorDigi::bookSummaryHisto(std::string recHitType) {
  
  std::string currentFolder = subsystemFolder_ +"/"+recHitType+"/"+ globalFolder_;
  dbe->setCurrentFolder(currentFolder);  
  
  MonitorElement * me = NULL;

  for(int r = 0; r < 3; r++){ //RPC regions are E-, B, and E+
    
    std::stringstream name;
    std::stringstream title;
    std::string regionName = RPCMonitorDigi::regionNames_[r];
    //Cluster size
    name<<"ClusterSize_"<< regionName;
    title<< "ClusterSize - "<<regionName;
    me = dbe->get(currentFolder+ "/" + name.str());
    if (me) dbe->removeElement(me->getName());
    ClusterSize_[r] = dbe->book1D(name.str(), title.str(),  20, 0.5, 20.5);
    
    //Number of Cluster
    name.str("");
    title.str("");
    name<<"NumberOfClusters_"<< regionName;
    title<< "Number of Clusters per Event - "<< regionName;
    me = dbe->get(currentFolder+ "/" + name.str());
    if (me) dbe->removeElement(me->getName());
    NumberOfClusters_[r] = dbe->book1D(name.str(), title.str(),  30, 0.5, 30.5);
    
    //Number of Digis
    name.str("");
    title.str("");
    name<<"Multiplicity_"<< regionName;
    title<< "Multiplicity per Event per Roll - "<< regionName;
    me = dbe->get(currentFolder+ "/" + name.str());
    if (me) dbe->removeElement(me->getName());
    NumberOfDigis_[r] = dbe->book1D(name.str(), title.str(), 50, 0.5, 50.5);

     
           
  }//end loop on regions

  me = dbe->get(currentFolder+ "/Occupancy_for_Endcap-");
  if (me) dbe->removeElement(me->getName());
  Occupancy_[EM] = dbe -> book2D("Occupancy_for_Endcap-", "Occupancy Endcap-", 6, 0.5 , 6.5, 4, -4.5, -0.5 );
  Occupancy_[EM]->setAxisTitle("Sec", 1);
  Occupancy_[EM]->setAxisTitle("Disk", 2);

  me = dbe->get(currentFolder+ "/Occupancy_for_Barrel");
  if (me) dbe->removeElement(me->getName());
  Occupancy_[B] = dbe -> book2D("Occupancy_for_Barrel", "Occupancy Barrel", 12, 0.5 , 12.5, 5, -2.5, 2.5 );
  Occupancy_[B]->setAxisTitle("Sec", 1);
  Occupancy_[B]->setAxisTitle("Wheel", 2);


  me = dbe->get(currentFolder+ "/Occupancy_for_Endcap+");
  if (me) dbe->removeElement(me->getName());
  Occupancy_[EP] = dbe -> book2D("Occupancy_for_Endcap+", "Occupancy Endcap+", 6, 0.5 , 6.5, 4, 0.5, 4.5 );
  Occupancy_[EP]->setAxisTitle("Sec", 1);
  Occupancy_[EP]->setAxisTitle("Disk", 2);
}
