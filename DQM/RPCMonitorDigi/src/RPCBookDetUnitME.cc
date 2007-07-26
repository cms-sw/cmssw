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
 sprintf(folder,"RPC/RecHits/%s/%s_%d/station_%d/sector_%d",regionName.c_str(),ringType.c_str(),
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
 sprintf(meId,"Occupancy_%s",detUnitLabel);
 sprintf(meTitle,"Occupancy_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

 sprintf(meId,"BXN_%s",detUnitLabel);
 sprintf(meTitle,"BXN_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 11, -10.5, 10.5);
 
 sprintf(meId,"BXN_vs_strip_%s",detUnitLabel);
 sprintf(meTitle,"BXN_vs_strip_for_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle,  100, 0.5, 100.5, 11, -10.5, 10.5);
 
 sprintf(meId,"ClusterSize_%s",detUnitLabel);
 sprintf(meTitle,"ClusterSize_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 21, 0.5, 20.5);
 
 sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
 sprintf(meTitle,"NumberOfClusters_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 10, 0.5, 10.5);
 
 sprintf(meId,"ClusterSize_vs_LowerStrip%s",detUnitLabel);
 sprintf(meTitle,"ClusterSize_vs_LowerStrip_for_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,11, 0.5, 11.5);
 
 sprintf(meId,"ClusterSize_vs_HigherStrip%s",detUnitLabel);
 sprintf(meTitle,"ClusterSize_vs_HigherStrip_for_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,11, 0.5, 11.5);
 
 sprintf(meId,"ClusterSize_vs_Strip%s",detUnitLabel);
 sprintf(meTitle,"ClusterSize_vs_Strip_for_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,11, 0.5, 11.5);
 
 sprintf(meId,"ClusterSize_vs_CentralStrip%s",detUnitLabel);
 sprintf(meTitle,"ClusterSize_vs_CentralStrip_for_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,11, 0.5, 11.5);
 
 sprintf(meId,"NumberOfDigi_%s",detUnitLabel);
 sprintf(meTitle,"NumberOfDigi_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 10, 0.5, 10.5);
 
 sprintf(meId,"CrossTalkLow_%s",detUnitLabel);
 sprintf(meTitle,"CrossTalkLow_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
 
 sprintf(meId,"CrossTalkHigh_%s",detUnitLabel);
 sprintf(meTitle,"CrossTalkHigh_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
 
 sprintf(meId,"BXWithData_%s",detUnitLabel);
 sprintf(meTitle,"NumberOfBXsWithData_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 5, 0.5, 5.5);
 
 
 /// RPCRecHits
 sprintf(meId,"MissingHits_%s",detUnitLabel);
 sprintf(meTitle,"MissingHits__for_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle, 100, 0, 100, 2, 0.,2.);

 sprintf(meId,"RecHitXPosition_%s",detUnitLabel);
 sprintf(meTitle,"RecHit_Xposition_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 80, -120, 120);
 
 sprintf(meId,"RecHitDX_%s",detUnitLabel);
 sprintf(meTitle,"RecHit_DX_for_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle, 30, -10, 10);
 
 sprintf(meId,"RecHitX_vs_dx_%s",detUnitLabel);
 sprintf(meTitle,"RecHit_Xposition_vs_Error_%s",layerLabel);
 meMap[meId] = dbe->book2D(meId, meTitle, 30, -100, 100,30,10,10);
 
 sprintf(meId,"RecHitCounter_%s",detUnitLabel);
 sprintf(meTitle,"RecHitCounter_%s",layerLabel);
 meMap[meId] = dbe->book1D(meId, meTitle,101,-0.5,100.5);
 

// sprintf(meId,"RecHitYPosition_%s",detUnitLabel);
// sprintf(meTitle,"RecHit_Yposition_for_%s",layerLabel);
// meMap[meId] = dbe->book1D(meId, meTitle, 40, -100, 100);
 
// sprintf(meId,"RecHitDY_%s",detUnitLabel);
// sprintf(meTitle,"RecHit_DY_for_%s",layerLabel);
// meMap[meId] = dbe->book1D(meId, meTitle, 30, -10, 10);
 
// sprintf(meId,"RecHitDXDY_%s",detUnitLabel);
// sprintf(meTitle,"RecHit_DXDY_for_%s",layerLabel);
// meMap[meId] = dbe->book1D(meId, meTitle, 30, -10, 10);

// sprintf(meId,"RecHitY_vs_dY_%s",detUnitLabel);
// sprintf(meTitle,"RecHit_Yposition_vs_Error_%s",layerLabel);
// meMap[meId] = dbe->book2D(meId, meTitle, 30, -100, 100,30,10,10);
 
	
	return meMap;
}










std::map<std::string, MonitorElement*> RPCMonitorDigi::bookRegionRing(int region, int ring) {

 std::map<std::string, MonitorElement*> meMap;
 
 std::string ringType = (region ==  0)?"Wheel":"Disk";
 
 dbe->setCurrentFolder(GlobalHistogramsFolder);

 char meId [128];
 char meTitle [128];
 
 sprintf(meId,"GlobalRecHitXYCoordinates_%s_%d_%d",ringType.c_str(),region,ring);
 sprintf(meTitle,"GlobalRecHitXYCoordinates_for_%s_%d_%d",ringType.c_str(),region,ring);
   meMap[meId] = dbe->book2D(meId, meTitle, 1000, -800, 800, 1000, -800, 800);
 
 return meMap;

}













