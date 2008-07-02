#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>


#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DQMOffline/Muon/interface/RPCEfficiency.h>
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
std::map<std::string, MonitorElement*> RPCEfficiency::bookDetUnitSeg(RPCDetId & detId,int nstrips) {
  
  std::map<std::string, MonitorElement*> meMap;
  std::string regionName;
  std::string ringType;
  if(detId.region()==0) {
    regionName="Barrel";
    ringType="Wheel";
  }
  else{
    ringType="Disk";
    if(detId.region() == -1) regionName="Endcap-";
    if(detId.region() ==  1) regionName="Endcap+";
  }
  
  char  folder[120];
  sprintf(folder,"RPC/MuonSegEff/%s/%s_%d/station_%d/sector_%d",regionName.c_str(),ringType.c_str(),detId.ring(),detId.station(),detId.sector());
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
    std::cout<<"Booking for the Barrel"<<detUnitLabel<<std::endl;
    
    sprintf(meId,"ExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
    sprintf(meId,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RealDetectedOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
    sprintf(meId,"RPCDataOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
    sprintf(meId,"BXDistribution_%s",detUnitLabel);
    sprintf(meTitle,"BXDistribution_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 11,-5, 5);
  }
  return meMap;
}



