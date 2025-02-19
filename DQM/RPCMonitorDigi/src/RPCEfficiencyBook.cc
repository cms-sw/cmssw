// #include <stdio.h>
// #include <stdlib.h>
// #include <iostream>
// #include <string>
// #include <map>


//#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DQM/RPCMonitorDigi/interface/RPCEfficiency.h>
//#include <DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h>
//#include "DQMServices/Core/interface/MonitorElement.h"

void RPCEfficiency::bookDetUnitSeg(RPCDetId & detId,int nstrips,std::string folder, std::map<std::string, MonitorElement*> & meMap) {
  
  //std::map<std::string, MonitorElement*> meMap;
   
  dbe->setCurrentFolder(folder);

  char meId [128];
  char meTitle [128];

  int rawId = detId.rawId();
    
  //Begin booking DT
  if(detId.region()==0) {
    
    sprintf(meId,"ExpectedOccupancyFromDT_%d",rawId);
    sprintf(meTitle,"ExpectedOccupancyFromDT_for_%d",rawId);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
    sprintf(meId,"RPCDataOccupancyFromDT_%d",rawId);
    sprintf(meTitle,"RPCDataOccupancyFromDT_for_%d",rawId);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
  //   sprintf(meId,"BXDistribution_%d",rawId);
//     sprintf(meTitle,"BXDistribution_for_%d",rawId);
//     meMap[meId] = dbe->book1D(meId, meTitle, 11,-5.5, 5.5);
  }else{
    //std::cout<<"Booking for the EndCap"<<detUnitLabel<<std::endl;

    sprintf(meId,"ExpectedOccupancyFromCSC_%d",rawId);
    sprintf(meTitle,"ExpectedOccupancyFromCSC_for_%d",rawId);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
    sprintf(meId,"RPCDataOccupancyFromCSC_%d",rawId);
    sprintf(meTitle,"RPCDataOccupancyFromCSC_for_%d",rawId);
    meMap[meId] = dbe->book1D(meId, meTitle, nstrips, 0.5, nstrips+0.5);
    
   //  sprintf(meId,"BXDistribution_%d",rawId);
//     sprintf(meTitle,"BXDistribution_for_%d",rawId);
//     meMap[meId] = dbe->book1D(meId, meTitle, 11,-5.5, 5.5);
  }
  //return meMap;
}



