/** \file
 * Impl of RPCDetId
 *
 * \author Ilaria Segoni
 * \version $Id: RPCDetId.cc,v 1.2 2005/10/27 10:28:39 segoni Exp $
 * \date 02 Aug 2005
 */

#include <iostream>
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
	      
	      
  if ( region     < minRegionId    || region    > maxRegionId ||
       ring       < minRingId      || ring      > maxRingId ||
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
  if(region == 0) ringInBits = ring + RingBarrelOffSet - minRingBarrelId;
  
  int stationInBits=station-minStationId;
  int sectorInBits=sector-minSectorId;
  int layerInBits=layer-minLayerId;
  int subSectorInBits=sector-minSubSectorId;
  int rollInBits=roll;
  
  id_ |= ( regionInBits    & RegionMask_)    << RegionStartBit_ | 
         ( ringInBits      & RingMask_)      << RingStartBit_  |
         ( stationInBits   & StationMask_)   << StationStartBit_ |
         ( sectorInBits    & SectorMask_)    << SectorStartBit_ |
         ( layerInBits     & LayerMask_)     << LayerStartBit_ |
         ( subSectorInBits & SubSectorMask_) << SubSectorStartBit_ |
         ( rollInBits      & RollMask_)      << RollStartBit_;
           

}


std::ostream& operator<<( std::ostream& os, const RPCDetId& id ){

// do differently whether it's station or Wheel.

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


