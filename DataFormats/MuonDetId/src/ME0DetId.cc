/** \file
 * Impl of ME0DetId
 */

#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h> 

ME0DetId::ME0DetId():DetId(DetId::Muon, MuonSubdetId::ME0){}


ME0DetId::ME0DetId(uint32_t id):DetId(id){
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::ME0) {
    throw cms::Exception("InvalidDetId") << "ME0DetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid ME0 id";  
  }
}

ME0DetId::ME0DetId(DetId id):DetId(id) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::ME0) {
    throw cms::Exception("InvalidDetId") << "ME0DetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid ME0 id";  
  }
}

ME0DetId::ME0DetId(int region, int layer,int chamber, int roll):	      
  DetId(DetId::Muon, MuonSubdetId::ME0)
{
  this->init(region,layer,chamber,roll);
}

void
ME0DetId::init(int region,int layer,int chamber,int roll)
{
  if ( region     < minRegionId    || region    > maxRegionId ||
       layer      < minLayerId     || layer     > maxLayerId ||
       chamber    < minChamberId   || chamber   > maxChamberId ||
       roll       < minRollId      || roll      > maxRollId) {
    throw cms::Exception("InvalidDetId") << "ME0DetId ctor:" 
					 << " Invalid parameters: " 
					 << " region "<<region
					 << " layer "<<layer
					 << " chamber "<<chamber
					 << " etaPartition "<<roll
					 << std::endl;
  }
  int regionInBits=region-minRegionId;
  int layerInBits=layer-minLayerId;
  int chamberInBits=chamber-minChamberId;
  int rollInBits=roll-minRollId;
  
  id_ |= ( regionInBits    & RegionMask_)    << RegionStartBit_    | 
         ( layerInBits     & LayerMask_)     << LayerStartBit_     |
         ( chamberInBits   & ChamberMask_)    << ChamberStartBit_  |
         ( rollInBits      & RollMask_)      << RollStartBit_        ;
   
}



std::ostream& operator<<( std::ostream& os, const ME0DetId& id ){


  os <<  " Region "<<id.region()
     << " Layer "<<id.layer()
     << " Chamber "<<id.chamber()
     << " EtaPartition "<<id.roll()
     <<" ";

  return os;
}


