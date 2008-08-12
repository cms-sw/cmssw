#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>


#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DQM/RPCMonitorDigi/interface/MuonSegmentEff.h>
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
std::map<std::string, MonitorElement*> MuonSegmentEff::bookDetUnitSeg(RPCDetId & detId) {
  
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
  //  if(detId.region()==0) {
    sprintf(meId,"ExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RealDetectedOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancy2DFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);
    
    sprintf(meId,"RPCDataOccupancyFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancyFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"RPCDataOccupancy2DFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancy2DFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);
    
    sprintf(meId,"RPCResidualsFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RPCResidualsFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 201,-100.5, 100.5);
    
    sprintf(meId,"RPCResiduals2DFromDT_%s",detUnitLabel);
    sprintf(meTitle,"RPCResiduals2DFromDT_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 201,-100.5, 100.5,200,-100.,100.);
    
    sprintf(meId,"EfficienyFromDTExtrapolation_%s",detUnitLabel);
    sprintf(meTitle,"EfficienyFromDTExtrapolation_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"EfficienyFromDT2DExtrapolation_%s",detUnitLabel);
    sprintf(meTitle,"EfficienyFromDT2DExtrapolation_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);

    //    }

    //if(detId.region()==-1 || detId.region()==1){
    //CSC
    sprintf(meId,"ExpectedOccupancyFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancyFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"RealDetectedOccupancyFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"RealDetectedOccupancyFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"ExpectedOccupancy2DFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"ExpectedOccupancy2DFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);
    
    sprintf(meId,"RPCDataOccupancyFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancyFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"RPCDataOccupancy2DFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"RPCDataOccupancy2DFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);
    
    sprintf(meId,"RPCResidualsFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"RPCResidualsFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 201,-100.5, 100.5);
    
    sprintf(meId,"RPCResiduals2DFromCSC_%s",detUnitLabel);
    sprintf(meTitle,"RPCResiduals2DFromCSC_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 201,-100.5, 100.5,200,-100.,100.);
    
    sprintf(meId,"EfficienyFromCSCExtrapolation_%s",detUnitLabel);
    sprintf(meTitle,"EfficienyFromCSCExtrapolation_for_%s",layerLabel);
    meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);
    
    sprintf(meId,"EfficienyFromCSC2DExtrapolation_%s",detUnitLabel);
    sprintf(meTitle,"EfficienyFromCSC2DExtrapolation_for_%s",layerLabel);
    meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);
    //}

  return meMap;
}



