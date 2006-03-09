#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>


#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>


/// Booking of MonitoringElemnt for one RPCDetId (= roll)

std::map<std::string, MonitorElement*> RPCMonitorDigi::bookDetUnitME(RPCDetId & detId) {
 
 
 std::map<std::string, MonitorElement*> meMap;
 
 std::string regionName;
 std::string ringType;
 if(detId.region() ==  0) {
 	regionName="Barrel";
 	ringType="Wheel";
 }else{
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
  



 sprintf(meId,"Oppupancy_%s",detUnitLabel);
 sprintf(meTitle,"Occupancy_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 100, 0, 100);

 sprintf(meId,"BXN_%s",detUnitLabel);
 sprintf(meTitle,"BXN_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 100, 0, 100);
 
 sprintf(meId,"ClusterSize_%s",detUnitLabel);
 sprintf(meTitle,"ClusterSize_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 11, 0, 11);
 
 sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
 sprintf(meTitle,"NumberOfClusters_or_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 10, 0, 10);
 
 sprintf(meId,"NumberOfDigi_%s",detUnitLabel);
 sprintf(meTitle,"NumberOfDigi_or_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 10000, 0, 10000);
 
	return meMap;
}


