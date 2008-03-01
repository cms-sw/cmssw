#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>
#include <sstream>

#include <DQM/L1TMonitor/interface/L1TRPCTPG.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace std;

/// Booking of MonitoringElemnt for one RPCDetId (= roll)






  std::map<std::string, MonitorElement*> 
   L1TRPCTPG::L1TRPCBookME(RPCDetId & detId) {
    
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
    sprintf(folder,"L1T/L1TRPCTPG/Strips/%s/%s_%d/station_%d/sector_%d",
     regionName.c_str(),ringType.c_str(),
     detId.ring(),detId.station(),detId.sector());
    //cout << folder << endl;
    dbe->setCurrentFolder(folder);
    
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
    
    
 // Begin booking
     sprintf(meId,"Occupancy_%s",detUnitLabel);
     sprintf(meTitle,"Occupancy_for_%s",layerLabel);
     //   cout << meId << endl;
     //   cout << meTitle << endl; 
     meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
      
     sprintf(meId,"BXN_%s",detUnitLabel);
     sprintf(meTitle,"BXN_for_%s",layerLabel);
     meMap[meId] = dbe->book1D(meId, meTitle, 11, -10.5, 10.5);
      
     sprintf(meId,"BXN_vs_strip_%s",detUnitLabel);
     sprintf(meTitle,"BXN_vs_strip_for_%s",layerLabel);
     meMap[meId] = dbe->book2D(meId, meTitle,  100, 0.5, 100.5, 11, -10.5, 10.5);
      
    return meMap;
  }
  
  



 
