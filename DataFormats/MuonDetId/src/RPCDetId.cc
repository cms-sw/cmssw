/** \file
 * Impl of RPCDetId
 *
 * \author Ilaria Segoni
 * \version $Id: RPCDetId.cc,v 1.27 2012/10/19 08:00:32 innocent Exp $
 * \date 02 Aug 2005
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h> 

#include<iostream>

RPCDetId::RPCDetId():DetId(DetId::Muon, MuonSubdetId::RPC),trind(0){}


RPCDetId::RPCDetId(uint32_t id):DetId(id),trind(0) {
  //  std::cout<<" constructor of the RPCDetId" <<std::endl;
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::RPC) {
    throw cms::Exception("InvalidDetId") << "RPCDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid RPC id";  
  }
}
RPCDetId::RPCDetId(DetId id):DetId(id),trind(0) {
  //  std::cout<<" constructor of the RPCDetId" <<std::endl;
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::RPC) {
    throw cms::Exception("InvalidDetId") << "RPCDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid RPC id";  
  }
}



RPCDetId::RPCDetId(int region, int ring, int station, int sector, int layer,int subsector, int roll):	      
  DetId(DetId::Muon, MuonSubdetId::RPC),trind(0)
{
  this->init(region,ring,station,sector,layer,subsector,roll);
}


void 
RPCDetId::buildfromDB(int region, int ring, int trlayer, int sector, 
		      const std::string& subs,
		      const std::string& roll,
		      const std::string& dbname){

  bool barrel = (region==0);
  //STATION
  int station = -1;
  if (barrel) {
    if (trlayer==1 || trlayer==2) station = 1;
    else if (trlayer==3 || trlayer==4) station = 2;
    else station = trlayer-2;
  } else {   
   station = abs(ring); 
  }


  //LAYER
  //int layer = 1;
  //if (barrel && station==1) layer = trlayer;
  //if (barrel && station==2) layer = trlayer-2; 

  //SUBSECTOR
  int subsector = 1;

  if (barrel) {
    if (station==3 && subs=="+") subsector = 2;
    if (station==4 && 
         (   sector==1 || sector==2 || sector==3 
                       || sector==5 || sector==6   
          || sector==7 || sector==8 
          || sector==10             || sector==12)
          && (subs=="+")) {
         subsector = 2;
    }

    if (station==4 && sector==4) {
      if (subs=="--") subsector=1;
      if (subs=="-")  subsector=2;
      if (subs=="+")  subsector=3;
      if (subs=="++") subsector=4;
    } 
  }

   // ROLL
  int iroll=0;

  if      (roll=="Backward" || roll=="A") iroll = 1;
  else if (roll=="Central" || roll=="B") iroll = 2;
  else if (roll=="Forward" || roll=="C") iroll = 3;
  else if (roll=="D") iroll = 4;
  else {
    std::cout << "** RPC: DBSpecToDetUnit, how to assigne roll to: "
         <<roll<<" ???" << std::endl;
  }

  int trIndex = 0;
  if(barrel){   
    //cout <<" BARREL: " << endl; 
    int eta_id = 6+ring;
    int plane_id = station;
    if(trlayer==2) plane_id=5;
    if(trlayer==4) plane_id=6;
    int sector_id = sector*3;
    int copy_id = subsector;
    int roll_id = iroll;
    trIndex=(eta_id*10000+plane_id*1000+sector_id*10+copy_id)*10+roll_id;
  } 
  else { 
    //    cout << "ENDCAP : " << endl;
    int eta_id = trlayer;
    if(ring>0) eta_id = 12-trlayer;
    int plane_id = abs(ring);
    int sector_id = sector;

    if (region <0){
      if (sector_id < 20 ){
	sector_id = 19+ 1-sector_id;
      }else{
	sector_id = 36+20-sector_id;
      }
    }
    sector_id-=1;

    //
    int copy_id = 1;
    int roll_id = iroll;
    trIndex=(eta_id*10000+plane_id*1000+sector_id*10+copy_id)*10+ roll_id;
  }
  this->buildfromTrIndex(trIndex);
}

void
RPCDetId::buildfromTrIndex(int trIndex)
{
  trind = trIndex;
  int eta_id = trIndex/100000;
  int region=0;
  int ring =0; 
  if (eta_id <=3 ){
    region = -1;
    ring = eta_id;
  }
  else if (eta_id >=9 ) {
    region = 1;
    ring = 12-eta_id;
  }
  else{
    region = 0;
    ring = eta_id - 6;
  }
  trIndex = trIndex%100000;
  int plane_id = trIndex/10000;
  int station=0;
  int layer=0;
  if (plane_id <=4){
    station = plane_id;
    layer = 1;
  }
  else{
    station = plane_id -4;
    layer = 2;
  }
  trIndex = trIndex%10000;
  int sector_id = trIndex/100;
  if (region!=0) {
        if ( !(ring == 1 && station > 1 && region==1)) {     
         sector_id+=1;
         if (sector_id==37)sector_id=1;
     }
  }
  if (region==-1){
    if (sector_id < 20 ){
      sector_id = 19+ 1-sector_id;
    }else{
      sector_id = 36+20-sector_id;
    }
  }
  trIndex = trIndex%100;
  int copy_id = trIndex/10;
  int sector=(sector_id-1)/3+1;
  if (region!=0) {
    sector=(sector+1)/2;
  }
  int subsector=0;
  if ( region == 0 ) {
    subsector = copy_id;
  }
  else {
    if ( ring == 1 && station > 1) {
      // 20 degree chambers
       subsector = ((sector_id+1)/2-1)%3+1;
    }else {
      // 10 degree chambers
      subsector = (sector_id-1)%6+1;
    }
//     std::cout <<" RE"<<station*region<<"/"<<ring<<" sector_id "<<sector_id
//    	      << " sector "<<sector <<" sub "<<subsector<<std::endl;
  }


  int roll=trIndex%10;
  this->init(region,ring,station,sector,layer,subsector,roll);
}



void
RPCDetId::init(int region,int ring,int station,int sector,
	       int layer,int subsector,int roll)
{
  int minRing=0;
  int maxRing=RPCDetId::maxRingForwardId;
  if (!region) 
    {
      minRing=RPCDetId::minRingBarrelId;
      maxRing=RPCDetId::maxRingBarrelId;
    } 
  
  if ( region     < minRegionId    || region    > maxRegionId ||
       ring       < minRing        || ring      > maxRing ||
       station    < minStationId   || station   > maxStationId ||
       sector     < minSectorId    || sector    > maxSectorId ||
       layer      < minLayerId     || layer     > maxLayerId ||
       subsector  < minSubSectorId || subsector > maxSubSectorId ||
       roll       < minRollId      || roll      > maxRollId) {
    throw cms::Exception("InvalidDetId") << "RPCDetId ctor:" 
					 << " Invalid parameters: " 
					 << " region "<<region
					 << " ring "<<ring
					 << " station "<<station
					 << " sector "<<sector
					 << " layer "<<layer
					 << " subsector "<<subsector
					 << " roll "<<roll
					 << std::endl;
  }
	      
  
  int regionInBits=region-minRegionId;
  int ringInBits =0;
  if(region != 0) ringInBits = ring - minRingForwardId;
  if(!region) ringInBits = ring + RingBarrelOffSet - minRingBarrelId;
  
  int stationInBits=station-minStationId;
  int sectorInBits=sector-(minSectorId+1);
  int layerInBits=layer-minLayerId;
  int subSectorInBits=subsector-(minSubSectorId+1);
  int rollInBits=roll;
  
  id_ |= ( regionInBits    & RegionMask_)    << RegionStartBit_    | 
         ( ringInBits      & RingMask_)      << RingStartBit_      |
         ( stationInBits   & StationMask_)   << StationStartBit_   |
         ( sectorInBits    & SectorMask_)    << SectorStartBit_    |
         ( layerInBits     & LayerMask_)     << LayerStartBit_     |
         ( subSectorInBits & SubSectorMask_) << SubSectorStartBit_ |
         ( rollInBits      & RollMask_)      << RollStartBit_        ;
   
}



std::ostream& operator<<( std::ostream& os, const RPCDetId& id ){


  os <<  " Re "<<id.region()
     << " Ri "<<id.ring()
     << " St "<<id.station()
     << " Se "<<id.sector()
     << " La "<<id.layer()
     << " Su "<<id.subsector()
     << " Ro "<<id.roll()
     << " Tr "<<id.trIndex()
     <<" ";

  return os;
}


