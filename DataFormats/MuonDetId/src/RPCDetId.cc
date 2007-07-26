/** \file
 * Impl of RPCDetId
 *
 * \author Ilaria Segoni
 * \version $Id: RPCDetId.cc,v 1.17 2006/09/05 17:31:38 mmaggi Exp $
 * \date 02 Aug 2005
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

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
      subsector = (sector_id-1)%3+1;
    }else {
      // 10 degree chambers
      subsector = (sector_id-1)%6+1;
    }
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


