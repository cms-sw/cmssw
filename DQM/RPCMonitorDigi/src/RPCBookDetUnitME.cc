#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>
#include <sstream>

#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include "DQMServices/Core/interface/MonitorElement.h"


using namespace std;

/// Booking of MonitoringElemnt for one RPCDetId (= roll)






  std::map<std::string, MonitorElement*> RPCMonitorDigi::bookDetUnitME(RPCDetId & detId) {
    
    // std::cout <<"Booking ME "<<detId<<std::endl; 
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
    
    char  folder[220];
    sprintf(folder,"RPC/RecHits/%s/%s_%d/station_%d/sector_%d",regionName.c_str(),ringType.c_str(),
	    detId.ring(),detId.station(),detId.sector());
    
    dbe->setCurrentFolder(folder);
    
    //temporary 
    
    char layer[128];
    sprintf(layer ,"layer_roll%d",detId.roll());
    std::cout<<"\n rool ->"<<layer<<std::endl;
   // sleep(20);

    /// Name components common to current RPDDetId  
    char detUnitLabel[328];
    char layerLabel[328];
    
  //sprintf(detUnitLabel ,"%d",detId());
    RPCGeomServ RPCname(detId);
    std::string nameRoll = RPCname.name();
    sprintf(detUnitLabel ,"%s",nameRoll.c_str());
    sprintf(layerLabel ,"%s",nameRoll.c_str());
   
    
    char meId [328];
    char meTitle [328];
    
        
    
    if (dqmexpert) {
      
      //std::string meId;
      
      /// BEgin booking
      sprintf(meId,"Occupancy_%s",detUnitLabel);
      sprintf(meTitle,"Occupancy_for_%s",layerLabel);
      

      /*
      std::cout <<"detUnitLabel:"<<detUnitLabel
		<<"\nLayerLabel "<<layerLabel
		<<"\nmeId "<<meId
	      <<"\nmeTitle "<<meTitle
		<<std::endl;
      */
      
      meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
      
      sprintf(meId,"BXN_%s",detUnitLabel);
      sprintf(meTitle,"BXN_for_%s",layerLabel);
      meMap[meId] = dbe->book1D(meId, meTitle, 21, -10.5, 10.5);
  }
    
    
    if (dqmsuperexpert) {
      
      
      sprintf(meId,"BXN_vs_strip_%s",detUnitLabel);
      sprintf(meTitle,"BXN_vs_strip_for_%s",layerLabel);
      meMap[meId] = dbe->book2D(meId, meTitle,  100, 0.5, 100.5, 21, -10.5, 10.5);
      
    }
  
    if (dqmexpert) {
      
      sprintf(meId,"ClusterSize_%s",detUnitLabel);
      sprintf(meTitle,"ClusterSize_for_%s",layerLabel);
      meMap[meId] = dbe->book1D(meId, meTitle, 20, 0.5, 20.5);

      /*
      std::cout <<"detUnitLabel:"<<detUnitLabel
		<<"\nLayerLabel "<<layerLabel
		<<"\nmeId "<<meId
		<<"\nmeTitle "<<meTitle
		<<std::endl;
      */

      sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
      sprintf(meTitle,"NumberOfClusters_for_%s",layerLabel);
      meMap[meId] = dbe->book1D(meId, meTitle, 10, 0.5, 10.5);
      
      /*
      std::cout <<"detUnitLabel:"<<detUnitLabel
		<<"\nLayerLabel "<<layerLabel
		<<"\nmeId "<<meId
		<<"\nmeTitle "<<meTitle
		<<std::endl;
      */
    }
    
    if (dqmsuperexpert) {
      
      sprintf(meId,"ClusterSize_vs_LowerStrip%s",detUnitLabel);
    
      //std::stringstream os;
      //os<<"ClusterSize_vs_LowerSrips%s"<<detUnitLabel;
      //meId +=os.str();
      
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
      
    }
    
    
    if (dqmexpert) {
      
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
      meMap[meId] = dbe->book1D(meId, meTitle, 11, -5.5, 5.5);
      
    }
    
    
    
    if (dqmsuperexpert) {
      
      /// RPCRecHits
      sprintf(meId,"MissingHits_%s",detUnitLabel);
      sprintf(meTitle,"MissingHits__for_%s",layerLabel);
      meMap[meId] = dbe->book2D(meId, meTitle, 100, 0, 100, 2, 0.,2.);
      
      sprintf(meId,"RecHitX_vs_dx_%s",detUnitLabel);
      sprintf(meTitle,"RecHit_Xposition_vs_Error_%s",layerLabel);
      meMap[meId] = dbe->book2D(meId, meTitle, 30, -100, 100,30,10,10);
      
    }
    
    if (dqmexpert) {
      
    sprintf(meId,"RecHitXPosition_%s",detUnitLabel);
    sprintf(meTitle,"RecHit_Xposition_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 80, -120, 120);
    
    sprintf(meId,"RecHitDX_%s",detUnitLabel);
    sprintf(meTitle,"RecHit_DX_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 30, -10, 10);
    
    
    
    sprintf(meId,"RecHitCounter_%s",detUnitLabel);
    sprintf(meTitle,"RecHitCounter_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle,100,-0.5,100.5);
    
    }
    
    
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
    
    
    // dbe->setCurrentFolder(GlobalHistogramsFolder);
    
    
    // sprintf(meId,"GlobalClusterSize_for_%s", regionName.c_str());
    // sprintf(meTitle,"GlobalClusterSize_for_%s", regionName.c_str());
    // meMap[meId] = dbe->book1D(meId, meTitle, 20, 0.5, 20.5);  
    
    //  sprintf(meId,"GlobalClusterSize_for_Endcap-");
    //sprintf(meTitle,"GlobalClusterSize_for_Enscap-");
    //meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);  
    
    //sprintf(meId,"GlobalClusterSize_for_Endcap+");
    //sprintf(meTitle,"GlobalClusterSize_for_Enscap+");
    //meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    dbe->setCurrentFolder(GlobalHistogramsFolder);
    sprintf(meId,"Occupancy_%s_%d_Sector_%d",ringType.c_str(),detId.ring(),detId.sector());
    sprintf(meTitle,"Occupancy_%s_%d_Sector_%d",ringType.c_str(),detId.ring(),detId.sector());

    if (detId.sector()==9 || detId.sector()==11 ) {
      
      meMap[meId] = dbe->book2D(meId, meTitle,  96, 0.5, 96.5, 14, 0.5, 14.5);
      
    }
    
    else  if (detId.sector()==4) {
      
      meMap[meId] = dbe->book2D(meId, meTitle,  96, 0.5, 96.5, 22, 0.5, 22.5);
    }
    else {
      meMap[meId] = dbe->book2D(meId, meTitle,  96, 0.5, 96.5, 16, 0.5, 16.5);
      
    }
    
    
    std::cout <<"End of Booking "<<std::endl;
    return meMap;
  }


  
  

  
std::map<std::string, MonitorElement*> RPCMonitorDigi::bookRegionRing(int region, int ring) {
  
  std::cout<<"begin of Global folder constructor"<<std::endl;;
  std::map<std::string, MonitorElement*> meMap;
  
  std::string ringType = (region ==  0)?"Wheel":"Disk";

  dbe->setCurrentFolder(GlobalHistogramsFolder);
  
  char meId [128];
  char meTitle [128];
  
  sprintf(meId,"GlobalRecHitXYCoordinates_%s_%d_%d",ringType.c_str(),region,ring);
  sprintf(meTitle,"GlobalRecHitXYCoordinates_for_%s_%d_%d",ringType.c_str(),region,ring);
  meMap[meId] = dbe->book2D(meId, meTitle, 1000, -800, 800, 1000, -800, 800);

  std::cout<<"end of Global folder"<<std::endl;;
  
  return meMap;
  
}


 
