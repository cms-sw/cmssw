#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <iostream>
#include <string>


uint32_t DBSpecToDetUnit::operator()(const ChamberLocationSpec & ch, 
   const FebLocationSpec & feb)
{
  //
  //FIXME !!!  semi-dummy and buggy method, check carefully !!!!
  //

  // REGION
  int region = -2;
  bool barrel = (ch.barrelOrEndcap=="Barrel");
  if (barrel) region = 0;
  else if (ch.diskOrWheel<0) region = -1;
  else if (ch.diskOrWheel>0) region = 1;

  // RING - what to hell is it.......
  int ring = ch.diskOrWheel;


  //STATION
  int station = -1;
  if (barrel) {
    if (ch.layer==1 || ch.layer==2) station = 1;
    else if (ch.layer==3 || ch.layer==4) station = 2;
    else station = ch.layer-2;
  } else {
   station = abs(ring); 
  }

  //LAYER
  int layer = 1;
  if (barrel && station==1) layer = ch.layer;
  if (barrel && station==2) layer = ch.layer-2; 

  //SECTOR
  int sector = ch.sector;

  //SUBSECTOR
  int subsector = 1;
  if (barrel) {
    if (station==3 && ch.subsector=="+") subsector = 2;
    if (station==4 && 
         (   sector==1 || sector==2 || sector==3 
                       || sector==5 || sector ==6   
          || sector==7 || sector==8 || sector==12)
          && (ch.subsector=="+"))          subsector = 2;
    if (station==4 && sector==4) {
      if (ch.subsector=="--") subsector=1;
      if (ch.subsector=="-")  subsector=2;
      if (ch.subsector=="+")  subsector=3;
      if (ch.subsector=="++") subsector=4;
    } 
  }


  // ROLL
  std::string roll = feb.cmsEtaPartition;
  int iroll=0;
  if      (roll=="1" || roll=="A") iroll = 1;
  else if (roll=="2" || roll=="B") iroll = 2;
  else if (roll=="3" || roll=="C") iroll = 3;
  else if (roll=="D") iroll = 4;
  else {
    std::cout << "** RPC: DBSpecToDetUnit, how to assigne roll to: "
               <<roll<<" ???" << std::endl;
  }

  RPCDetId du(region, ring, station, sector, layer, subsector, iroll);
  return du.rawId();
}
