/**********************************************
 *                                            *
 *           Giuseppe Roselli                 *
 *         INFN, Sezione di Bari              *
 *      Via Amendola 173, 70126 Bari          *
 *         Phone: +390805443218               *
 *      giuseppe.roselli@ba.infn.it           *
 *                                            *
 *                                            *
 **********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <map>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DQM/RPCMonitorDigi/interface/RPCEfficiencyFromTrack.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"

std::map<std::string, MonitorElement*> RPCEfficiencyFromTrack::bookDetUnitTrackEff(RPCDetId & detId, const edm::EventSetup & iSetup) {
  
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
  sprintf(folder,"RPC/EfficiencyFromTrack/%s/%s_%d/station_%d/sector_%d",regionName.c_str(),ringType.c_str(),
	  detId.ring(),detId.station(),detId.sector());

  dbe->setCurrentFolder(folder);


  int strips = 0; double lastvalue = 0.;




  edm::ESHandle<RPCGeometry> rpcgeo;
  iSetup.get<MuonGeometryRecord>().get(rpcgeo);

  const RPCRoll * rpcRoll = rpcgeo->roll(detId);
  strips = rpcRoll->nstrips();

  if(strips == 0 ) strips = 1;
  lastvalue=(double)strips+0.5;


  RPCGeomServ RPCname(detId);
  std::string nameRoll = RPCname.name();


  char detUnitLabel[128];
  char layerLabel[128];

  sprintf(detUnitLabel ,"%s",nameRoll.c_str());
  sprintf(layerLabel ,"%s",nameRoll.c_str());
  
  char meId [128];
  char meTitle [128];
  
  //Begin booking
  sprintf(meId,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
  sprintf(meTitle,"ExpectedOccupancyFromTrack_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, strips, 0.5, lastvalue);

  sprintf(meId,"RPCDataOccupancy_%s",detUnitLabel);
  sprintf(meTitle,"RPCDataOccupancy_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, strips, 0.5, lastvalue);
  
  sprintf(meId,"Residuals_%s",detUnitLabel);
  sprintf(meTitle,"Residuals_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, 150,-49.5, 49.);

  sprintf(meId,"EfficienyFromTrackExtrapolation_%s",detUnitLabel);
  sprintf(meTitle,"EfficienyFromTrackExtrapolation_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle, strips, 0.5, lastvalue);

  sprintf(meId,"ClusterSize_%s",detUnitLabel);
  sprintf(meTitle,"ClusterSize_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle,10,0.5,10.5);

  sprintf(meId,"BunchX_%s",detUnitLabel);
  sprintf(meTitle,"BunchX_for_%s",layerLabel);
  meMap[meId] = dbe->book1D(meId, meTitle,13,-6.5,6.5);

  return meMap;
}
