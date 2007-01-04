#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <iostream>
#include <string>

using namespace std;

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
                       || sector==5 || sector==6   
          || sector==7 || sector==8 
          || sector==10             || sector==12)
          && (ch.subsector=="+"))          subsector = 2;
    if (station==4 && sector==4) {
      if (ch.subsector=="--") subsector=1;
      if (ch.subsector=="-")  subsector=2;
      if (ch.subsector=="+")  subsector=3;
      if (ch.subsector=="++") subsector=4;
    } 
  }
   

  // ROLL
  string roll = feb.cmsEtaPartition;
  int iroll=0;
  if      (roll=="1" || roll=="A") iroll = 1;
  else if (roll=="2" || roll=="B") iroll = 2;
  else if (roll=="3" || roll=="C") iroll = 3;
  else if (roll=="D") iroll = 4;
  else {
    cout << "** RPC: DBSpecToDetUnit, how to assigne roll to: "
               <<roll<<" ???" << endl;
  }

/*
  RPCDetId du;
  bool doDebug = false;
  try {
   du = RPCDetId(region, ring, station, sector, layer, subsector, iroll);
  } 
  catch(...) {
    doDebug = true;
    cout <<" Problem with RPCDetId, got exception!! " <<endl;
  }
*/

  RPCDetId du2;
  if(barrel){   
    //cout <<" BARREL: " << endl; 
    int eta_id = 6+ch.diskOrWheel;
    int plane_id = station;
    if(ch.layer==2) plane_id=5;
    if(ch.layer==4) plane_id=6;
    int sector_id = ch.sector*3;
    int copy_id = subsector;
    int roll_id = iroll;
    int trIndex=(eta_id*10000+plane_id*1000+sector_id*10+copy_id)*10+roll_id;
    //cout<<"DBSpec: "<<trIndex;
    du2.buildfromTrIndex(trIndex);
  } 
  else { 
    //cout << "ENDCAP : " << endl;
    int eta_id = ch.layer;
    if(ch.diskOrWheel>0) eta_id = 12-ch.layer;
    int plane_id = abs(ch.diskOrWheel);
    int sector_id = ch.sector;
    int copy_id = 1;
    int roll_id = iroll;
    int trIndex=(eta_id*10000+plane_id*1000+sector_id*10+copy_id)*10+
      roll_id;
    //cout<<"DBSpec: "<<trIndex;
    du2.buildfromTrIndex(trIndex);
  }

/*
  if (doDebug || du != du2) {
    cout << "barrel: " << barrel << endl;
    cout << ch.print(); 
    cout <<" detId: "<<du2<<", rawId:"<<du2()<<endl;
    cout <<" way (1) "<< endl
         <<" region: "<<region
         <<" ring: " << ring
         <<" station: "<<station
         <<" sector: "<<sector
         <<" layer: "<<layer
         <<" subsector: "<<subsector
         <<" iroll: "<<iroll
         << endl; 
    cout <<" way(2) " <<endl
         <<" region: "<<du2.region()
         <<" ring: " << du2.ring()
         <<" station: "<<du2.station()
         <<" sector: "<<du2.sector()
         <<" layer: "<<du2.layer()
         <<" subsector: "<<du2.subsector()
         <<" iroll: "<<du2.roll()
         << endl; 
    return du2();
  }
*/
  return du2.rawId();
}
