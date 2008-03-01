#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>


//#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>
#include <DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DQMServices/Core/interface/MonitorElement.h"

/// Booking of MonitoringElemnt for one RPCDetId (= roll)

std::map<std::string, MonitorElement*> RPCMonitorEfficiency::bookDetUnitMEEff(RPCDetId & detId) {
  
  std::map<std::string, MonitorElement*> meMap;
  std::string regionName;
  std::string ringType;
  if(detId.region()==0) {
    regionName="Barrel";
    ringType="Wheel";
  }
  else{
    ringType="Disk";
    if(detId.region() == -1) regionName="Encap-";
    if(detId.region() ==  1) regionName="Encap+";
  }
  
  char  folder[120];
  sprintf(folder,"RPC/Efficiency/%s/%s_%d/station_%d/sector_%d",regionName.c_str(),ringType.c_str(),
	  detId.ring(),detId.station(),detId.sector());
  
  //std::cout<<"BOOKING 1"<<std::endl;
  dbe->setCurrentFolder(folder);
  //std::cout<<"BOOKING 2"<<std::endl;
  /// Name components common to current RPDDetId  
  char detUnitLabel[128];
  char layerLabel[128];
  sprintf(detUnitLabel ,"%d",detId());
  sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());
  //std::cout<<"BOOKING 3"<<std::endl;
  
  char meId [128];
  char meTitle [128];
  
  // Begin booking
  sprintf(meId,"ExpectedOccupancyFromDT_%s",detUnitLabel);
  sprintf(meTitle,"ExpectedOccupancyFromDT_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"ExpectedOccupancyFromDT_forCrT_%s",detUnitLabel);
  sprintf(meTitle,"ExpectedOccupancyFromDT_forCrT_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"YExpectedOccupancyFromDT_%s",detUnitLabel);
  sprintf(meTitle,"YExpectedOccupancyFromDT_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, -100, 100);

  sprintf(meId,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
  sprintf(meTitle,"ExpectedOccupancy2DFromDT_for_%s",layerLabel);
  meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);

  //std::cout<<"BOOKING 4"<<std::endl;
  sprintf(meId,"RPCDataOccupancy_%s",detUnitLabel);
  sprintf(meTitle,"RPCDataOccupancy_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"RPCDataOccupancy2D_%s",detUnitLabel);
  sprintf(meTitle,"RPCDataOccupancy2D_for_%s",layerLabel);
  meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);

  sprintf(meId,"RealDetectedOccupancy_%s",detUnitLabel);
  sprintf(meTitle,"RealDetectedOccupancy_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  //std::cout<<"BOOKING 5"<<std::endl;
  sprintf(meId,"EfficienyFromDTExtrapolation_%s",detUnitLabel);
  sprintf(meTitle,"EfficienyFromDTExtrapolation_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"EfficienyFromDT2DExtrapolation_%s",detUnitLabel);
  sprintf(meTitle,"EfficienyFromDT2DExtrapolation_for_%s",layerLabel);
  meMap[meId] = dbe->book2D(meId, meTitle, 100, 0.5, 100.5,200,-100.,100.);

  sprintf(meId,"XCrossTalkFromDTExtrapolation_1_%s",detUnitLabel);
  sprintf(meTitle,"XCrossTalkFromDTExtrapolation_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"XCrossTalkFromDTExtrapolation_2_%s",detUnitLabel);
  sprintf(meTitle,"XCrossTalkFromDTExtrapolation_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"XCrossTalkFromDetectedStrip_1_%s",detUnitLabel);
  sprintf(meTitle,"XCrossTalkFromDetectedStrip_1_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"XCrossTalkFromDetectedStrip_2_%s",detUnitLabel);
  sprintf(meTitle,"XCrossTalkFromDetectedStrip_2_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"YCrossTalkFromDTExtrapolation_1_%s",detUnitLabel);
  sprintf(meTitle,"YCrossTalkFromDTExtrapolation_1_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 200, -100., 100.);

  sprintf(meId,"YCrossTalkFromDTExtrapolation_2_%s",detUnitLabel);
  sprintf(meTitle,"YCrossTalkFromDTExtrapolation_2_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 200, -100., 100.);

  sprintf(meId,"RPCResiduals_%s",detUnitLabel);
  sprintf(meTitle,"RPCResiduals_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 201,-100.5, 100.5);

  sprintf(meId,"RPCResiduals2D_%s",detUnitLabel);
  sprintf(meTitle,"RPCResiduals2D_for_%s",layerLabel);
  meMap[meId] = dbe->book2D(meId, meTitle, 201,-100.5, 100.5,200,-100.,100.);

  sprintf(meId,"XCrossTalk_1_%s",detUnitLabel);
  sprintf(meTitle,"XCrossTalk_1_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"XCrossTalk_2_%s",detUnitLabel);
  sprintf(meTitle,"XCrossTalk_2_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"XDetectCrossTalk_1_%s",detUnitLabel);
  sprintf(meTitle,"XDetectCrossTalk_1_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"XDetectCrossTalk_2_%s",detUnitLabel);
  sprintf(meTitle,"XDetectCrossTalk_2_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, 0.5, 100.5);

  sprintf(meId,"YCrossTalk_1_%s",detUnitLabel);
  sprintf(meTitle,"YCrossTalk_1_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 100, -100, 100);

  sprintf(meId,"YCrossTalk_2_%s",detUnitLabel);
  sprintf(meTitle,"YCrossTalk_2_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 200, -100., 100.);

  return meMap;
}
