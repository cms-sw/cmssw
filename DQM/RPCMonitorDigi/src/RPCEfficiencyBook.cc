#include <DQM/RPCMonitorDigi/interface/RPCEfficiency.h>


void RPCEfficiency::bookDetUnitSeg(RPCDetId & detId,int nstrips,std::string folder, std::map<std::string, MonitorElement*> & meMap) {
  
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



