/** \file
 * Impl of GEMDetId
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h> 

GEMDetId::GEMDetId():DetId(DetId::Muon, MuonSubdetId::GEM){}


GEMDetId::GEMDetId(uint32_t id):DetId(id){
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::GEM) {
    throw cms::Exception("InvalidDetId") << "GEMDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid GEM id";  
  }
}

GEMDetId::GEMDetId(DetId id):DetId(id) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::GEM) {
    throw cms::Exception("InvalidDetId") << "GEMDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid GEM id";  
  }
}

GEMDetId::GEMDetId(int region, int ring, int station, int layer,int chamber, int roll):	      
  DetId(DetId::Muon, MuonSubdetId::GEM)
{
  this->init(region,ring,station,layer,chamber,roll);
}

void
GEMDetId::init(int region,int ring,int station,
	       int layer,int chamber,int roll)
{
  if ( region     < minRegionId    || region    > maxRegionId ||
       ring       < minRingId      || ring      > maxRingId ||
       station    < minStationId   || station   > maxStationId ||
       layer      < minLayerId     || layer     > maxLayerId ||
       chamber    < minChamberId   || chamber   > maxChamberId ||
       roll       < minRollId      || roll      > maxRollId) {
    throw cms::Exception("InvalidDetId") << "GEMDetId ctor:" 
					 << " Invalid parameters: " 
					 << " region "<<region
					 << " ring "<<ring
					 << " station "<<station
					 << " layer "<<layer
					 << " chamber "<<chamber
					 << " roll "<<roll
					 << std::endl;
  }
  int regionInBits=region-minRegionId;
  int ringInBits = ring-minRingId;
  int stationInBits=station-minStationId;
  int layerInBits=layer-minLayerId;
  int chamberInBits=chamber-(minChamberId+1);
  int rollInBits=roll;
  
  id_ |= ( regionInBits    & RegionMask_)    << RegionStartBit_    | 
         ( ringInBits      & RingMask_)      << RingStartBit_      |
         ( stationInBits   & StationMask_)   << StationStartBit_   |
         ( layerInBits     & LayerMask_)     << LayerStartBit_     |
         ( chamberInBits   & ChamberMask_)    << ChamberStartBit_  |
         ( rollInBits      & RollMask_)      << RollStartBit_        ;
   
}



std::ostream& operator<<( std::ostream& os, const GEMDetId& id ){


  os <<  " Re "<<id.region()
     << " Ri "<<id.ring()
     << " St "<<id.station()
     << " La "<<id.layer()
     << " Ch "<<id.chamber()
     << " Ro "<<id.roll()
     <<" ";

  return os;
}


