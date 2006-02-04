/** \file
 * Impl of RPCDetId
 *
 * \author Ilaria Segoni
 * \version $Id: RPCDetId.cc,v 1.5 2005/11/01 15:19:03 segoni Exp $
 * \date 02 Aug 2005
 */

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

RPCDetId::RPCDetId():DetId(DetId::Muon, MuonSubdetId::RPC){}


RPCDetId::RPCDetId(uint32_t id):DetId(id) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::RPC) {
    throw cms::Exception("InvalidDetId") << "RPCDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid RPC id";  
  }
}



RPCDetId::RPCDetId(int region, int ring, int station, int sector, int layer,int subsector, int roll):	      
        DetId(DetId::Muon, MuonSubdetId::RPC)
{
  this->init(region,ring,station,sector,layer,subsector,roll);
}


void
RPCDetId::buildfromTrIndex(int trIndex)
{
  int region=0;
  int ring=0;
  int station=0;
  int sector=0;
  int layer=0;
  int subsector=0;
  int roll=0;
  this->init(region,ring,station,sector,layer,subsector,roll);
}



void
RPCDetId::init(int region,int ring,int station,int sector,
	       int layer,int subsector,int roll)
{
  int minRing=RPCDetId::minRingForwardId;
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
  int sectorInBits=sector-minSectorId;
  int layerInBits=layer-minLayerId;
  int subSectorInBits=subsector-minSubSectorId;
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
     <<" ";

  return os;
}


