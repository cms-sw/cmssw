#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>


EcalElectronicsId::EcalElectronicsId() {
  EcalElectronicsId_=0xFFFFFFFFu;
}

EcalElectronicsId::EcalElectronicsId(uint32_t id) {
  EcalElectronicsId_=id;
}

EcalElectronicsId::EcalElectronicsId(int dccid, int towerid, int channelid){
  if ( (dccid < MIN_DCCID) || (dccid > MAX_DCCID) ||  
       (towerid < MIN_TOWERID) || (towerid > MAX_TOWERID) ||  
       (channelid < MIN_CHANNELID) || (channelid > MAX_CHANNELID) )
    throw cms::Exception("InvalidDetId") << "EcalElectronicsId:  Cannot create object.  Indexes out of bounds.";
  EcalElectronicsId_= (channelid&0x1F) | ( (towerid&0x7F) << 5) | ((dccid&0x7F) << 12);
}


std::ostream& operator<<(std::ostream& os,const EcalElectronicsId& id) 
{
  return os << id.dccId() << ',' << id.towerId() << ',' << id.channelId() ;
}

