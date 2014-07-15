#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DQM/RPCMonitorClient/interface/RPCEfficiencySecond.h>
#include <DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

std::map<std::string, MonitorElement*> RPCEfficiencySecond::bookDetUnitSeg(RPCDetId & detId,int nstrips, std::string folderPath) {
  
  std::map<std::string, MonitorElement*> meMap;
   
  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure(); 

  std::string folder = folderPath+ "RollByRoll/" +  folderStr->folderStructure(detId);

  delete folderStr;

  dbe->setCurrentFolder(folder);

  RPCGeomServ RPCname(detId);
  std::string nameRoll = RPCname.name();
  char detUnitLabel[128];
  char layerLabel[128];

  sprintf(detUnitLabel ,"%s",nameRoll.c_str());
  sprintf(layerLabel ,"%s",nameRoll.c_str());

  char meId [128];
  char meTitle [128];
  
  //Begin booking DT
  if(detId.region()==0) {

    //std::cout<<"Booking "<<folder<<meId<<std::endl;
    sprintf(meId,"ExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    //std::cout<<"Booking "<<meId<<std::endl;
 
    sprintf(meId,"RPCDataOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);

    sprintf(meId,"Profile_%s",detUnitLabel);
    sprintf(meTitle,"Profile_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    //std::cout<<"Booking "<<folder<<meId<<std::endl;

//     sprintf(meId,"BXDistribution_%s",detUnitLabel);
//     sprintf(meTitle,"BXDistribution_for_%s",layerLabel);
//     meMap[meId] = dbe->book1D(meId, meTitle, 11,-5.5, 5.5);
    
  }else{
    //std::cout<<"Booking for the EndCap"<<detUnitLabel<<std::endl;

    //std::cout<<"Booking "<<meId<<std::endl;
    sprintf(meId,"ExpectedOccupancyFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancyFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
       
    //std::cout<<"Booking "<<meId<<std::endl;
    sprintf(meId,"RPCDataOccupancyFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancyFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
    //std::cout<<"Booking "<<meId<<std::endl;
    sprintf(meId,"Profile_%s",detUnitLabel);
    sprintf(meTitle,"Profile_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
//     //std::cout<<"Booking "<<meId<<std::endl;
//     sprintf(meId,"BXDistribution_%s",detUnitLabel);
//     sprintf(meTitle,"BXDistribution_for_%s",layerLabel);
//     meMap[meId] = dbe->book1D(meId, meTitle, 11,-5.5, 5.5);
  }
  return meMap;
}



